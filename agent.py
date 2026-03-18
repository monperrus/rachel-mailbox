"""
Autonomous Email Agent
- Polls a dedicated IMAP inbox
- Uses LangGraph to reason and decide on a reply
- Sends replies via SMTP automatically
"""

import imaplib
import re
import smtplib
import email
import email.utils
import ssl
import time
import uuid
import logging
import argparse
import importlib.util
import json
import urllib.request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Config and llm are initialised by load_config() before run() is called.
Config = None
llm = None


def load_config(path: str) -> None:
    """Load a config file by path and initialise Config + llm."""
    global Config, llm
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Config = module.Config
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        api_key=Config.ANTHROPIC_AUTH_TOKEN,
        base_url=Config.ANTHROPIC_BASE_URL + "/v1" if Config.ANTHROPIC_BASE_URL else None,
    )
    log.info(f"Loaded config from {path}")


# ── State ──────────────────────────────────────────────────────────────────────

class EmailState(TypedDict):
    uid: str
    sender: str
    reply_to: str         # Reply-To header of the incoming email (may be empty)
    cc: str               # CC header of the incoming email (may be empty)
    subject: str
    body: str
    message_id: str       # Message-ID of the incoming email
    references: str       # References header of the incoming email (space-separated IDs)
    thread_history: str
    is_auto_reply: bool
    should_reply: bool
    reply_body: str
    error: str


# ── Document Context ─────────────────────────────────────────────────────────

def fetch_document_context(subject: str, body: str) -> str:
    """
    If Config.DOCUMENTS is defined, ask the LLM which documents are relevant to
    this email, fetch their URLs, and return the combined content as a string
    ready to be injected into the reply prompt.

    Config.DOCUMENTS format:
        [{"name":"<short-name>", "description": "<human-readable description>", "url": "<url>"}]
    """
    documents = getattr(Config, "DOCUMENTS", None)
    if not documents or len(documents) == 0:
        log.debug("[docs] No documents configured, skipping.")
        return ""

    descriptions_list = json.dumps(documents)
    log.debug(f"[docs] Selecting from {len(documents)} document(s) for subject={subject!r}")
    selector_prompt = (
        f"You are a document relevance selector.\n"
        f"An email agent received the following email:\n"
        f"SUBJECT: {subject}\n"
        f"BODY:\n{body}\n\n"
        f"Available reference documents:\n{descriptions_list}\n\n"
        f"Return ONLY a JSON array of document names for the documents "
        f"that would help answer this email. Return an empty array [] if none are relevant. "
        f"Example: [\"doc1\", \"doc3\"]"
    )
    try:
        selection_result = llm.invoke([
            SystemMessage(content="You select relevant reference documents for an email agent. Output only valid JSON."),
            HumanMessage(content=selector_prompt),
        ])
        raw = selection_result.content.strip()
        log.debug(f"[docs] Selector LLM raw response: {raw!r}")
        # Strip markdown code fences if present
        raw = re.sub(r"^```[^\n]*\n?(.*?)```$", r"\1", raw, flags=re.DOTALL).strip()
        selected_indices = json.loads(raw)
        if not isinstance(selected_indices, list):
            log.debug(f"[docs] Selector returned non-list: {selected_indices!r}, ignoring.")
            selected_indices = []
    except Exception as e:
        log.warning(f"Document selector LLM call failed: {e}")
        return ""

    log.debug(f"[docs] Selected indices/names: {selected_indices}")
    if not selected_indices:
        log.debug("[docs] No relevant documents selected.")
        return ""

    fetched_sections = []
    for doc_name in selected_indices:
        doc = next((d for d in documents if d["name"] == doc_name), None)
        if not doc:
            log.debug(f"[docs] No document found with name: {doc_name!r}")
            continue
        url = doc["url"]
        description = doc["description"]
        log.debug(f"[docs] Fetching '{description}' from {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AI-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            log.info(f"[docs] Fetched '{description}' from {url} ({len(content)} chars)")
            fetched_sections.append(f"--- Reference: {description} ({url}) ---\n{content}")
        except Exception as e:
            log.warning(f"[docs] Failed to fetch '{description}' from {url}: {e}")

    if not fetched_sections:
        log.debug("[docs] All fetches failed or returned nothing.")
        return ""

    log.debug(f"[docs] Returning {len(fetched_sections)} document section(s).")
    return "\n\n".join(fetched_sections)


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def triage(state: EmailState) -> EmailState:
    """Decide whether this email warrants a reply based on auto-reply headers."""
    state["should_reply"] = not state["is_auto_reply"]
    log.info(f"Triage for '{state['subject']}': {'REPLY' if state['should_reply'] else 'SKIP (auto-reply)'}")
    return state


def generate_reply(state: EmailState) -> EmailState:
    """Generate a reply to the email."""
    if not state["should_reply"]:
        return state

    thread_section = ""
    if state["thread_history"]:
        thread_section = f"\nPrevious thread context:\n{state['thread_history']}\n"

    doc_context = fetch_document_context(state["subject"], state["body"])
    doc_section = ""
    if doc_context:
        doc_section = f"\nRelevant reference documents:\n{doc_context}\n"

    prompt = f"""
{thread_section}
{doc_section}
You received this email:
FROM: {state['sender']}
SUBJECT: {state['subject']}
BODY:
{state['body']}

Write a helpful, professional reply. Be concise. Do not use filler phrases like 
"I hope this email finds you well." Sign off as: {Config.AGENT_NAME}

Write only the email body, no subject line."""
    
    print(prompt)

    result = llm.invoke([
        SystemMessage(content=Config.AGENT_PERSONA),
        HumanMessage(content=prompt)
    ])
    state["reply_body"] = result.content.strip()
    return state


def send_reply(state: EmailState) -> EmailState:
    """Send the generated reply via SMTP and copy it to the sent folder via IMAP."""
    if not state["should_reply"] or not state["reply_body"]:
        return state

    try:
        # Reply-to-all: prefer Reply-To over From for the primary recipient
        primary = state["reply_to"] or state["sender"]
        all_recipients = [primary]
        if state["cc"]:
            all_recipients.append(state["cc"])

        msg = MIMEMultipart()
        msg["Message-ID"] = email.utils.make_msgid(domain=Config.SMTP_HOST)
        msg["Date"] = email.utils.formatdate(localtime=True)
        msg["From"] = Config.EMAIL_ADDRESS
        msg["To"] = primary
        if state["cc"]:
            msg["CC"] = state["cc"]
        msg["Subject"] = f"Re: {state['subject']}"
        msg["Auto-Submitted"] = "auto-replied"
        # Thread headers: link this reply into the conversation chain
        if state["message_id"]:
            msg["In-Reply-To"] = state["message_id"]
            # References = existing chain + the message we're replying to
            prior_refs = state["references"].strip() if state["references"] else ""
            new_refs = (prior_refs + " " + state["message_id"]).strip()
            msg["References"] = new_refs
        msg.attach(MIMEText(state["reply_body"], "plain"))
        raw = msg.as_bytes()

        with smtplib.SMTP_SSL(Config.SMTP_HOST, Config.SMTP_PORT) as server:
            server.login(Config.EMAIL_ADDRESS, Config.EMAIL_PASSWORD)
            server.sendmail(Config.EMAIL_ADDRESS, all_recipients, raw)

        log.info(f"Replied to {state['sender']} re: '{state['subject']}'")

        # Copy to sent folder if one was detected at startup
        if len(_folders_to_search) > 1:
            sent_folder = _folders_to_search[1]
            try:
                ctx = ssl.create_default_context()
                with imaplib.IMAP4_SSL(Config.IMAP_HOST, Config.IMAP_PORT, ssl_context=ctx) as imap:
                    imap.login(Config.EMAIL_ADDRESS, Config.EMAIL_PASSWORD)
                    imap.append(sent_folder, "\\Seen", imaplib.Time2Internaldate(time.time()), raw)
                log.info(f"Copied reply to {sent_folder}")
            except Exception as e:
                log.warning(f"Failed to copy reply to sent folder: {e}")

    except Exception as e:
        state["error"] = str(e)
        log.error(f"Failed to send reply: {e}")

    return state


def route_after_triage(state: EmailState) -> str:
    return "generate_reply" if state["should_reply"] else END


# ── Build Graph ────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(EmailState)
    graph.add_node("triage", triage)
    graph.add_node("generate_reply", generate_reply)
    graph.add_node("send_reply", send_reply)

    graph.set_entry_point("triage")
    graph.add_conditional_edges("triage", route_after_triage)
    graph.add_edge("generate_reply", "send_reply")
    graph.add_edge("send_reply", END)

    return graph.compile()


agent = build_graph()


# ── IMAP Helpers ───────────────────────────────────────────────────────────────

def decode_str(value):
    if not value:
        return ""
    parts = decode_header(value)
    return "".join(
        part.decode(enc or "utf-8") if isinstance(part, bytes) else part
        for part, enc in parts
    )


def get_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode("utf-8", errors="replace")
    else:
        return msg.get_payload(decode=True).decode("utf-8", errors="replace")
    return ""


AUTO_REPLY_HEADERS = {
    "auto-submitted",
    "x-auto-response-suppress",
    "x-autoreply",
    "x-autorespond",
}
AUTO_REPLY_PRECEDENCE = {"bulk", "list", "auto_reply", "junk"}


def is_auto_reply_email(msg) -> bool:
    """Return True if the message contains standard auto-reply headers."""
    auto_submitted = (msg.get("Auto-Submitted") or "").strip().lower()
    if auto_submitted and auto_submitted != "no":
        return True
    precedence = (msg.get("Precedence") or "").strip().lower()
    if precedence in AUTO_REPLY_PRECEDENCE:
        return True
    for h in ("X-Auto-Response-Suppress", "X-Autoreply", "X-Autorespond"):
        if msg.get(h) is not None:
            return True
    return False


# Common sent-folder candidates to probe at startup
_SENT_FOLDER_CANDIDATES = ["Sent", "Sent Items", "Sent Messages", "[Gmail]/Sent Mail", "INBOX.Sent", "SENT"]

# Resolved at startup; populated by detect_folders()
_folders_to_search: list[str] = ["INBOX"]


def detect_folders(imap) -> None:
    """Probe the server once to find which sent folder exists and cache it."""
    global _folders_to_search
    for candidate in _SENT_FOLDER_CANDIDATES:
        try:
            status, _ = imap.select(candidate, readonly=True)
            if status == "OK":
                log.info(f"Sent folder detected: {candidate}")
                _folders_to_search = ["INBOX", candidate]
                imap.select("INBOX")
                return
        except Exception:
            continue
    log.warning("No sent folder detected; thread history will only search INBOX.")
    _folders_to_search = ["INBOX"]


def fetch_thread_history(imap, msg, max_messages: int = 10, max_body_chars: int = 800) -> str:
    """
    Reconstruct the full conversation thread by following References + In-Reply-To
    headers recursively, so the original message is always included.
    """
    # Collect all IDs to look up: References chain + In-Reply-To
    references = msg.get("References", "") or ""
    in_reply_to = msg.get("In-Reply-To", "") or ""
    log.debug(f"[thread] References: {references!r}")
    log.debug(f"[thread] In-Reply-To: {in_reply_to!r}")

    # Start with the explicitly listed references
    all_refs = re.findall(r"<[^>]+>", references + " " + in_reply_to)
    if not all_refs:
        log.debug("[thread] No message-id references found; skipping thread fetch.")
        return ""

    # Deduplicate, keep order
    seen_ids: set = set()
    queue = [r for r in all_refs if not (r in seen_ids or seen_ids.add(r))]
    log.debug(f"[thread] Folders to search: {_folders_to_search}")

    history_msgs: list[tuple[str, str, str]] = []  # (date, from, body) for sorting

    # BFS: fetch each referenced message; if it has further References, enqueue those too
    processed: set = set()
    while queue and len(history_msgs) < max_messages:
        msg_id = queue.pop(0)
        if msg_id in processed:
            continue
        processed.add(msg_id)

        found = False
        for folder in _folders_to_search:
            try:
                status, data = imap.select(folder, readonly=True)
                log.debug(f"[thread] SELECT {folder!r} → {status}")
                if status != "OK":
                    continue
                _, uids = imap.search(None, "HEADER", "Message-ID", msg_id.strip("<>"))
                log.debug(f"[thread] SEARCH {msg_id!r} in {folder!r} → {uids}")
                if not uids[0]:
                    continue
                uid = uids[0].split()[-1]
                _, data = imap.fetch(uid, "(RFC822)")
                raw = data[0][1]
                prev = email.message_from_bytes(raw)
                from_ = decode_str(prev["From"])
                date_ = prev.get("Date", "")
                body_ = (get_body(prev) or "")[:max_body_chars]
                log.debug(f"[thread] Found {msg_id!r} from {from_!r} in {folder!r}")
                history_msgs.append((date_, from_, body_))

                # Enqueue any further ancestors this message references
                prev_refs = re.findall(r"<[^>]+>", (prev.get("References") or "") + " " + (prev.get("In-Reply-To") or ""))
                for ref in prev_refs:
                    if ref not in seen_ids:
                        seen_ids.add(ref)
                        queue.append(ref)

                found = True
                break
            except Exception as exc:
                log.debug(f"[thread] Error searching {folder!r} for {msg_id!r}: {exc}")
                continue
        if not found:
            log.debug(f"[thread] {msg_id!r} not found in any folder.")

    # Restore INBOX for the caller
    imap.select("INBOX")

    # Sort by Date ascending so the prompt reads oldest → newest
    def _parse_date(d):
        try:
            return email.utils.parsedate_to_datetime(d)
        except Exception:
            return None

    history_msgs.sort(key=lambda t: (_parse_date(t[0]) is None, _parse_date(t[0])))

    parts = [f"--- From: {f} | Date: {d}\n{b}" for d, f, b in history_msgs]
    log.debug(f"[thread] Returning {len(parts)} historical message(s).")
    return "\n\n".join(parts)


def fetch_unseen_emails(imap):
    imap.select("INBOX")
    _, uids = imap.search(None, "UNSEEN")
    emails = []
    for uid in uids[0].split():
        _, data = imap.fetch(uid, "(RFC822)")
        raw = data[0][1]
        msg = email.message_from_bytes(raw)
        thread_history = fetch_thread_history(imap, msg)
        incoming_msg_id = (msg.get("Message-ID") or "").strip()
        incoming_refs   = (msg.get("References") or "").strip()
        emails.append({
            "uid": uid.decode(),
            "sender": decode_str(msg["From"]),
            "reply_to": decode_str(msg["Reply-To"]),
            "cc": decode_str(msg["CC"]),
            "subject": decode_str(msg["Subject"]),
            "body": get_body(msg),
            "message_id": incoming_msg_id,
            "references": incoming_refs,
            "thread_history": thread_history,
            "is_auto_reply": is_auto_reply_email(msg),
        })
        # Mark as seen
        imap.store(uid, "+FLAGS", "\\Seen")
    return emails


# ── Main Loop ──────────────────────────────────────────────────────────────────

def run():
    log.info(f"Agent starting. Monitoring {Config.EMAIL_ADDRESS}")
    ctx = ssl.create_default_context()
    with imaplib.IMAP4_SSL(Config.IMAP_HOST, Config.IMAP_PORT, ssl_context=ctx) as imap:
        imap.login(Config.EMAIL_ADDRESS, Config.EMAIL_PASSWORD)
        detect_folders(imap)
    while True:
        try:
            ctx = ssl.create_default_context()
            with imaplib.IMAP4_SSL(Config.IMAP_HOST, Config.IMAP_PORT, ssl_context=ctx) as imap:
                imap.login(Config.EMAIL_ADDRESS, Config.EMAIL_PASSWORD)
                emails = fetch_unseen_emails(imap)

            log.info(f"Found {len(emails)} unseen email(s)")
            for e in emails:
                state: EmailState = {
                    "uid": e["uid"],
                    "sender": e["sender"],
                    "reply_to": e["reply_to"],
                    "cc": e["cc"],
                    "subject": e["subject"],
                    "body": e["body"],
                    "message_id": e["message_id"],
                    "references": e["references"],
                    "thread_history": e["thread_history"],
                    "is_auto_reply": e["is_auto_reply"],
                    "should_reply": False,
                    "reply_body": "",
                    "error": "",
                }
                agent.invoke(state)

        except Exception as e:
            log.error(f"Poll error: {e}")

        time.sleep(Config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autonomous email agent")
    parser.add_argument(
        "--config",
        default="config.py",
        help="Path to config file (default: config.py)",
    )
    args = parser.parse_args()
    load_config(args.config)
    run()
