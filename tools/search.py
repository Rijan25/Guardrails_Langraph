"""
Search tool: wraps Tavily's web search API with a mock fallback.

If TAVILY_API_KEY is not set, the mock returns plausible-looking placeholder
results so the pipeline can be developed and demoed without spending API credits.
"""

import os
from typing import Optional


async def search_web(query: str) -> list[str]:
    """Run a web search for `query`. Returns a list of result snippets."""
    api_key = os.getenv("TAVILY_API_KEY")

    if api_key:
        return await _tavily_search(query, api_key)
    else:
        print(f"  [search] No TAVILY_API_KEY found — using mock results for: {query!r}")
        return _mock_search(query)


async def _tavily_search(query: str, api_key: str) -> list[str]:
    """Call the real Tavily API and return result snippets."""
    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=api_key)
        response = await client.search(query, max_results=3)
        results = response.get("results", [])
        return [r.get("content", "") for r in results if r.get("content")]
    except ImportError:
        print("  [search] tavily-python not installed — falling back to mock")
        return _mock_search(query)
    except Exception as e:
        print(f"  [search] Tavily error: {e} — falling back to mock")
        return _mock_search(query)


def _mock_search(query: str) -> list[str]:
    """Return deterministic mock snippets for offline/demo use."""
    return [
        f"[Mock Result 1] According to recent sources, {query} has seen significant activity "
        f"in 2024, with multiple breakthroughs reported by leading research institutions.",
        f"[Mock Result 2] Experts in the field note that {query} continues to evolve rapidly. "
        f"Key developments include advances in methodology and increased funding from both "
        f"public and private sectors.",
        f"[Mock Result 3] A comprehensive review of {query} highlights challenges and "
        f"opportunities. Practitioners recommend staying current with the latest peer-reviewed "
        f"literature for the most accurate perspective.",
    ]
