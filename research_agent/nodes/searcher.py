"""
Searcher Node: executes web searches for each sub-task in the plan.

WHY THIS NODE EXISTS:
  The planner produces a list of focused sub-questions; this node executes a
  web search for each one using Tavily and collects the raw result snippets.
  These snippets are passed to the summarizer which synthesizes them into a
  final coherent answer.

  NeMo is NOT placed here — search results are intermediate data from the web,
  not user-facing output. Validating raw snippets with NeMo would be noisy and
  wasteful; the output rail on the summarizer catches any issues in the final answer.
"""

import asyncio

from research_agent.state import ResearchState
from research_agent.tools.search import search_web


async def searcher_node(state: ResearchState) -> dict:
    """
    Searches the web for each sub-task in the plan concurrently.
    Returns all snippets concatenated into the 'search_results' list.
    """
    plan = state.get("plan", [])
    if not plan:
        print("[searcher] No plan found — skipping search")
        return {"search_results": []}

    print(f"\n[searcher] Running {len(plan)} searches concurrently...")

    # Run all searches in parallel for speed
    tasks = [search_web(sub_query) for sub_query in plan]
    results_per_task = await asyncio.gather(*tasks)

    all_results: list[str] = []
    for sub_query, snippets in zip(plan, results_per_task):
        print(f"  [searcher] '{sub_query}' -> {len(snippets)} results")
        for snippet in snippets:
            if snippet.strip():
                all_results.append(f"[Query: {sub_query}]\n{snippet}")

    print(f"[searcher] Total snippets collected: {len(all_results)}")
    return {"search_results": all_results}
