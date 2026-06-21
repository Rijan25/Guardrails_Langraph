"""
Summarizer Node: synthesizes search results into a final answer, then validates
the output with NeMo output rails before returning it.

WHY NeMo IS PLACED HERE:
  This is the last node before the answer reaches the user, making it the ideal
  place for output validation. We use rails.check_async() with both the user
  message and assistant response — NeMo auto-detects this as an output rail check
  and runs the self_check_output flow from guardrails/output/config.yml.

  If the result is BLOCKED or MODIFIED with a refusal, we return NeMo's safe
  replacement instead of the raw LLM output.
"""

import os
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.options import RailStatus, RailType

from research_agent.state import ResearchState

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OUTPUT_CONFIG_DIR = Path(__file__).parent.parent / "guardrails" / "output"


async def summarizer_node(state: ResearchState) -> dict:
    """
    Synthesizes search results with Groq Llama, then validates the output
    with NeMo output rails before writing to state['summary'].
    """
    query = state["query"]
    search_results = state.get("search_results", [])

    print(f"\n[summarizer] Synthesizing answer for: {query!r}")
    print(f"[summarizer] Using {len(search_results)} search snippets")

    raw_summary = await _generate_summary(query, search_results)
    print(f"[summarizer] Raw summary generated ({len(raw_summary)} chars)")

    validated_summary = await _apply_output_rails(query, raw_summary)
    print(f"[summarizer] Output rails check complete")

    return {"summary": validated_summary}


async def _generate_summary(query: str, search_results: list[str]) -> str:
    """Call Groq Llama to synthesize the search snippets into a coherent answer."""
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    context = "\n\n".join(search_results) if search_results else "No search results available."

    system_prompt = """You are a research assistant. Synthesize the provided search results
into a clear, accurate, and well-structured summary that directly answers the research question.
Be factual and balanced. If the search results are mock/placeholder data, still produce a
coherent summary based on your general knowledge of the topic."""

    user_prompt = f"""Research Question: {query}

Search Results:
{context}

Please provide a comprehensive summary that answers the research question."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    return response.content.strip()


async def _apply_output_rails(query: str, summary: str) -> str:
    """
    Run NeMo check_async() with user + assistant messages so it auto-runs output rails.
    Returns the original summary if safe, or NeMo's replacement if blocked/modified.
    """
    print("[summarizer] Applying NeMo output rails...")

    try:
        config = RailsConfig.from_path(str(OUTPUT_CONFIG_DIR))
        llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
        rails = LLMRails(config, llm=llm)

        # Passing both user + assistant triggers output rail auto-detection
        result = await rails.check_async(
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": summary},
            ],
            rail_types=[RailType.OUTPUT],
        )

        if result.status == RailStatus.BLOCKED:
            print(f"[summarizer] NeMo blocked the output (rail: {result.rail})")
            return "I was unable to generate a safe summary for this query."

        if result.status == RailStatus.MODIFIED and result.content:
            print("[summarizer] NeMo modified the output")
            return result.content

        return summary

    except Exception as e:
        print(f"[summarizer] NeMo output rail error: {e} — using raw summary")
        return summary
