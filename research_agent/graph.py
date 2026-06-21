"""
Graph Assembly: wires all nodes into the LangGraph StateGraph.

Flow:
  input_guard → [conditional] → planner → searcher → summarizer → END
                              ↘ END (if unsafe)

The conditional edge after input_guard routes based on state["is_safe"]:
  - True  → "planner"
  - False → END (short-circuits the pipeline immediately)
"""

from langgraph.graph import StateGraph, END

from research_agent.state import ResearchState
from research_agent.nodes.input_guard import input_guard_node
from research_agent.nodes.planner import planner_node
from research_agent.nodes.searcher import searcher_node
from research_agent.nodes.summarizer import summarizer_node


def route_after_guard(state: ResearchState) -> str:
    """
    Routing function called after input_guard completes.
    Returns "safe" or "unsafe" — mapped to destinations in add_conditional_edges.
    """
    if state.get("is_safe", False):
        return "safe"
    return "unsafe"


def build_graph() -> StateGraph:
    """Construct and compile the research assistant graph."""
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("summarizer", summarizer_node)

    # Entry point
    graph.set_entry_point("input_guard")

    # Conditional routing after safety check:
    #   "safe"   → planner (continue pipeline)
    #   "unsafe" → END     (block immediately, no further nodes run)
    graph.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {
            "safe": "planner",
            "unsafe": END,
        },
    )

    # Linear edges for the happy path
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()
