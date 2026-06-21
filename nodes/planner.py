"""
Planner Node: breaks the user query into 2-3 focused sub-tasks.

WHY THIS NODE EXISTS:
  A single broad query like "What are the latest developments in quantum computing?"
  is hard to search effectively in one shot. The planner uses a Groq Llama call to
  decompose it into specific sub-questions that the searcher can tackle individually,
  improving the breadth and relevance of retrieved information.

  NeMo is NOT placed here — the input has already been validated by input_guard,
  and the output of this node (a list of search queries) is never shown to the user
  directly, so output validation isn't needed at this step.
"""

import os
import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from state import ResearchState

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


async def planner_node(state: ResearchState) -> dict:
    """
    Uses Groq Llama to decompose the query into 2-3 search sub-tasks.
    Returns a list of sub-task strings in the 'plan' field.
    """
    query = state["query"]
    print(f"\n[planner] Planning research for: {query!r}")

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    system_prompt = """You are a research planning assistant. Given a research question,
break it down into 2-3 specific, focused sub-questions that can be searched independently.
Each sub-question should target a distinct aspect of the main topic.

Respond with a JSON array of strings. Example:
["What is X?", "How does Y work?", "What are the recent advances in Z?"]

Return ONLY the JSON array — no explanation, no markdown."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research question: {query}"),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content.strip()

    plan = _parse_plan(raw, query)
    print(f"[planner] Generated plan with {len(plan)} sub-tasks:")
    for i, task in enumerate(plan, 1):
        print(f"  {i}. {task}")

    return {"plan": plan}


def _parse_plan(raw: str, fallback_query: str) -> list[str]:
    """Parse LLM JSON output; fall back to a single-item plan if parsing fails."""
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        plan = json.loads(raw.strip())
        if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
            return plan[:3]  # cap at 3 sub-tasks
    except Exception as e:
        print(f"[planner] Failed to parse plan JSON ({e}), using query as single task")
    return [fallback_query]
