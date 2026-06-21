"""
ResearchState: The shared state schema passed between all LangGraph nodes.

Each node reads from and writes to this TypedDict. LangGraph uses it to track
the full lifecycle of a research query — from raw input through planning,
searching, and final summarization.
"""

from typing import TypedDict


class ResearchState(TypedDict):
    query: str               # original user query
    plan: list[str]          # planner's list of sub-tasks
    search_results: list[str]  # raw search results from searcher node
    summary: str             # final synthesized answer
    intent: str              # detected by NeMo input rails (e.g. "ask research question")
    is_safe: bool            # True if NeMo input rails pass the query
