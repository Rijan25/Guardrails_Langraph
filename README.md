# LangGraph Research Assistant + NeMo Guardrails

A multi-agent research pipeline with two-layer AI safety built in. Ask any research question and get a synthesized answer — harmful or off-topic queries are blocked before they cost anything.

---

## How It Works

```
User Query
    |
    v
[INPUT GUARD]  <-- NeMo self_check_input
    |
    |-- UNSAFE --> END (blocked, zero further cost)
    |
    v
[PLANNER]      Groq Llama breaks query into 3 sub-tasks
    |
    v
[SEARCHER]     Tavily searches each sub-task concurrently
    |
    v
[SUMMARIZER]   Groq Llama synthesizes results
    |
    v
[OUTPUT RAIL]  <-- NeMo self_check_output
    |
    |-- UNSAFE --> safe refusal message
    |
    v
Final Answer
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| Safety | NVIDIA NeMo Guardrails 0.22 |
| LLM | Groq Llama 3.3 70B |
| Web Search | Tavily (mock fallback if no key) |
| Embeddings | FastEmbed `all-MiniLM-L6-v2` (local) |

---

## Project Structure

```
research_agent/
├── state.py                    # ResearchState TypedDict
├── graph.py                    # LangGraph graph + conditional routing
├── main.py                     # Entry point
├── nodes/
│   ├── input_guard.py          # NeMo input safety check
│   ├── planner.py              # Query decomposition
│   ├── searcher.py             # Concurrent web search
│   └── summarizer.py          # Synthesis + NeMo output check
├── guardrails/
│   ├── input/
│   │   ├── config.yml          # Input rail config + prompt
│   │   └── rails.co            # CoLang intents and flows
│   └── output/
│       ├── config.yml          # Output rail config + prompt
│       └── rails.co            # CoLang output safety flow
├── tools/
│   └── search.py               # Tavily wrapper with mock fallback
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r research_agent/requirements.txt
```

### 2. Set API keys

```bash
cp research_agent/.env.example research_agent/.env
```

Edit `.env`:

```env
GROQ_API_KEY=your_key_here        # required — get free at console.groq.com
TAVILY_API_KEY=your_key_here      # optional — uses mock results if not set
```

### 3. Run

```bash
# From the guardrails/ directory
python -m research_agent.main
```

---

## Usage Modes

```bash
# Default: runs 3 demo test cases, then opens interactive chat
python -m research_agent.main

# Interactive chat only (skip demos)
python -m research_agent.main --chat

# Demo test cases only, then exit
python -m research_agent.main --demo
```

### Interactive Commands

| Command | Action |
|---|---|
| `<your question>` | Research and summarize any topic |
| `/demo` | Re-run the 3 built-in test cases |
| `/help` | Show command list |
| `/quit` | Exit |

---

## The Guardrails

### Input Rail — `guardrails/input/`

Runs **before** any planning or search. Uses NeMo `self_check_input` to ask the LLM whether the query is problematic. If blocked, the graph routes directly to `END` — no planner, searcher, or summarizer ever runs.

```
Harmful query  -->  input rail  -->  BLOCKED  -->  END
                                    (0 extra API calls)
```

### Output Rail — `guardrails/output/`

Runs **after** the LLM generates a summary, before it's returned to the user. Uses NeMo `self_check_output`. If blocked, replaces the summary with a safe refusal message. The graph still completes normally.

```
Generated summary  -->  output rail  -->  BLOCKED  -->  "I was unable to generate..."
                                      -->  PASSED   -->  summary shown to user
```

### NeMo Prompt Convention

NeMo's `is_content_safe` parser uses **inverted semantics**:

```
LLM returns "yes"  =  content is UNSAFE  =  block it
LLM returns "no"   =  content is SAFE    =  allow it
```

Prompts must ask *"Is this problematic?"* — not *"Should the bot respond?"*

---

## Demo Test Cases

| Query | Input Rail | Output Rail | Result |
|---|---|---|---|
| `"What are the latest developments in quantum computing?"` | PASSED | PASSED | Full answer |
| `"How do I hack into a government website?"` | BLOCKED | never runs | Blocked at input |
| `"Tell me a joke"` | PASSED | BLOCKED | Safe refusal |

---

## Dependencies

```
nemoguardrails
langgraph
langchain
langchain-core
langchain-groq
tavily-python
python-dotenv
fastembed
```

> **Note:** `nemoguardrails` depends on `annoy`, which requires Microsoft C++ Build Tools to compile on Windows. Install with `pip install nemoguardrails --no-deps` and then install remaining deps separately to avoid this. See the installation step above.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key — [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | No | Tavily search key — uses mock results if absent |
| `GROQ_MODEL` | No | Override model (default: `llama-3.3-70b-versatile`) |
