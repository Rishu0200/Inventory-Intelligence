"""
LangGraph StateGraph definition.
Defines the full agentic workflow:
  router → [demand | reorder | supplier | anomaly] agent → synthesiser → END
"""
from __future__ import annotations
from typing import TypedDict
from langgraph.graph import StateGraph, END
from orchestrator.llm_factory import get_llm
from orchestrator.router import classify_intent
from config import settings



class InventoryState(TypedDict):
    query:          str   # User's original query
    intent:         str   # Classified route: demand/reorder/supplier/anomaly/general
    sku_id:         str   # Extracted SKU (empty string if not found)
    rag_context:    str   # Retrieved document chunks
    tool_result:    str   # Raw output from the agent's tools
    final_response: str   # Synthesised, user-facing answer



def router_node(state: InventoryState) -> dict:
    intent, sku_id = classify_intent(state["query"])
    return {"intent": intent, "sku_id": sku_id}


def route_selector(state: InventoryState) -> str:
    """Conditional edge function — maps intent to next node name."""
    intent = state.get("intent", "general")
    mapping = {
        "demand":   "demand_agent",
        "reorder":  "reorder_agent",
        "supplier": "supplier_agent",
        "anomaly":  "anomaly_agent",
        "general":  "demand_agent",   # fallback
    }
    return mapping.get(intent, "demand_agent")



def synthesiser_node(state: InventoryState) -> dict:
    """
    Combine tool_result + rag_context into a coherent final response.
    Uses LLM if available; otherwise returns tool_result directly.
    """
    tool_result = state.get("tool_result", "No result.")
    rag_context = state.get("rag_context", "")

    if not settings.use_llm:
        return {"final_response": tool_result}

    try:
        llm = get_llm(temperature=0.3, max_tokens=512)
        context_snippet = rag_context[:600] if rag_context else "No additional context."
        prompt = (
            "You are an inventory intelligence assistant for Uninox Houseware, "
            "a Delhi-based kitchenware manufacturer.\n\n"
            f"User query: {state['query']}\n\n"
            f"Analysis result:\n{tool_result}\n\n"
            f"Supporting document context:\n{context_snippet}\n\n"
            "Give a clear, concise, actionable response. "
            "Use specific numbers from the analysis. Be direct."
        )
        resp = llm.invoke(prompt)
        return {"final_response": resp.content.strip()}
    except Exception as exc:
        return {"final_response": f"{tool_result}\n\n[LLM synthesis failed: {exc}]"}


# ── Import agent nodes (lazy to avoid circular imports) ──────────────────────

def _load_agents():
    from agents.demand_agent   import demand_agent_node
    from agents.reorder_agent  import reorder_agent_node
    from agents.supplier_agent import supplier_agent_node
    from agents.anomaly_agent  import anomaly_agent_node
    return (demand_agent_node, reorder_agent_node,
            supplier_agent_node, anomaly_agent_node)


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    (demand_fn, reorder_fn,
     supplier_fn, anomaly_fn) = _load_agents()

    workflow = StateGraph(InventoryState)

    # Add nodes
    workflow.add_node("router",         router_node)
    workflow.add_node("demand_agent",   demand_fn)
    workflow.add_node("reorder_agent",  reorder_fn)
    workflow.add_node("supplier_agent", supplier_fn)
    workflow.add_node("anomaly_agent",  anomaly_fn)
    workflow.add_node("synthesiser",    synthesiser_node)

    # Entry point
    workflow.set_entry_point("router")

    # Conditional routing from router
    workflow.add_conditional_edges(
        "router",
        route_selector,
        {
            "demand_agent":   "demand_agent",
            "reorder_agent":  "reorder_agent",
            "supplier_agent": "supplier_agent",
            "anomaly_agent":  "anomaly_agent",
        },
    )

    # All agents flow to synthesiser
    for agent in ["demand_agent", "reorder_agent", "supplier_agent", "anomaly_agent"]:
        workflow.add_edge(agent, "synthesiser")

    workflow.add_edge("synthesiser", END)

    return workflow



_graph = None

def get_graph():
    """Return compiled graph (singleton)."""
    global _graph
    if _graph is None:
        _graph = build_graph().compile()
    return _graph


def run_query(query: str) -> str:
    """
    Run a query through the full agentic pipeline.
    Returns the final response string.
    """
    graph = get_graph()
    initial_state: InventoryState = {
        "query":          query,
        "intent":         "",
        "sku_id":         "",
        "rag_context":    "",
        "tool_result":    "",
        "final_response": "",
    }
    result = graph.invoke(initial_state)
    return result.get("final_response", "Unable to process query.")
