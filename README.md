# AI Agent over email

This is a fully autonomous AI agent that monitors a dedicated inbox and replies to emails without human intervention. You set a persona, point it at a mailbox, and walk away. It handles everything else — including the emails you were never going to answer anyway.

It has been designed to handle the mailbox of [AI Scientist Rachel So](https://project-rachel.4open.science/).

For a commercial version of this, see <https://usemajordomo.com/>.

## Architecture

```
IMAP poll → triage (LLM) → generate reply (LLM) → send via SMTP
```

Uses **LangGraph** for the agent loop and **Claude** as the LLM. No database. No UI. No drama.

## Setup

### 1. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure
Copy `config.py.template` and fill in your values, or set environment variables:

```bash
export EMAIL_ADDRESS="agent@yourdomain.com"
export EMAIL_PASSWORD="your-app-password"
export IMAP_HOST="imap.gmail.com"
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="465"
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_BASE_URL="https://yourfavorite"
export LLM_MODEL="yourmodel"
export AGENT_NAME="Alex"
export AGENT_PERSONA="a helpful assistant for Acme Corp..."
export POLL_INTERVAL_SECONDS="60"
```

### 4. Gmail setup (if using Gmail)
- Create a dedicated Gmail account for the agent
- Enable 2FA and generate an **App Password**
- Use the App Password as `EMAIL_PASSWORD`

### 5. Run
```bash
python agent.py
```

## How it works

1. **Poll** — checks for UNSEEN emails every N seconds via IMAP
2. **Triage** — decides if the email warrants a reply (filters spam, auto-replies, newsletters, and the guy who keeps emailing about his invoice)
3. **Generate** — writes a reply in your persona and tone
4. **Send** — sends via SMTP and marks the original as Seen, as if it were never a problem

