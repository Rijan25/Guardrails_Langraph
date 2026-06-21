"""
Input Guard Node: the first node in the graph.

WHY NeMo IS PLACED HERE:
  This node acts as a gatekeeper before any expensive LLM work begins.
  We use rails.check_async() — NeMo's dedicated safety-check API — to run
  the self_check_input flow from guardrails/input/config.yml against the
  raw user query.

  check_async() returns a RailsResult with status PASSED or BLOCKED.
  If BLOCKED, is_safe=False and the graph routes to END immediately —
  no planner or searcher calls are ever made.

  Intent is detected via a separate lightweight LLM call so we can label
  the state even when NeMo blocks the request.
"""

import os
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.options import RailStatus, RailType

from state import ResearchState

INPUT_CONFIG_DIR = Path(__file__).parent.parent / "guardrails" / "input"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


async def input_guard_node(state: ResearchState) -> dict:
    """
    Runs NeMo input rails + intent detection against the user query.
    Returns updates to is_safe and intent fields.
    """
    query = state["query"]
    print(f"\n[input_guard] Checking query: {query!r}")

    # Run both in sequence: intent first (cheap), then NeMo safety check
    intent = await _detect_intent(query)
    print(f"[input_guard] Intent detected: {intent!r}")

    is_safe = await _run_nemo_input_check(query)
    print(f"[input_guard] Safe: {is_safe}")

    return {"intent": intent, "is_safe": is_safe}


async def _detect_intent(query: str) -> str:
    """Classify the query intent with a direct LLM call."""
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""Classify the following user message into exactly one category:
- "ask research question": the user wants factual information or research on a topic
- "express greeting": the user is saying hello or making small talk
- "ask off topic": the user is asking for something unrelated to research (jokes, games, poems, etc.)
- "ask harmful": the user is asking for instructions to do something illegal or harmful

User message: "{query}"

Respond with only the category name, nothing else."""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()

    if "greeting" in raw:
        return "express greeting"
    if "off topic" in raw or "off-topic" in raw:
        return "ask off topic"
    if "harmful" in raw:
        return "ask harmful"
    return "ask research question"


async def _run_nemo_input_check(query: str) -> bool:
    """
    Use NeMo's check_async() to run the self_check_input rail.
    Returns True if the query passes (safe), False if blocked.
    """
    try:
        config = RailsConfig.from_path(str(INPUT_CONFIG_DIR))
        llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
        rails = LLMRails(config, llm=llm)

        # check_async with only a user message auto-runs input rails
        result = await rails.check_async(
            messages=[{"role": "user", "content": query}],
            rail_types=[RailType.INPUT],
        )

        if result.status == RailStatus.BLOCKED:
            print(f"[input_guard] NeMo blocked query (rail: {result.rail})")
            return False

        return True

    except Exception as e:
        print(f"[input_guard] NeMo check error: {e} — falling back to intent-based check")
        # If NeMo fails, block anything that isn't a research question
        intent = await _detect_intent(query)
        return intent == "ask research question"
