"""
POST /api/query — Natural language query endpoint.
Runs the LangGraph orchestrator and returns the final response.
"""
from __future__ import annotations
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import QueryRequest, QueryResponse
from orchestrator.graph import run_query, get_graph

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest):
    """
    Ask any inventory question in natural language.

    Examples:
    - "Forecast demand for TBP-001 next 3 months"
    - "Which SKUs need reordering?"
    - "Who is the best supplier for RSH-001?"
    - "Are there any demand anomalies this quarter?"
    """
    graph = get_graph()

    initial_state = {
        "query":          body.question,
        "intent":         "",
        "sku_id":         "",
        "rag_context":    "",
        "tool_result":    "",
        "final_response": "",
    }

    result = graph.invoke(initial_state)

    return QueryResponse(
        question=body.question,
        intent=result.get("intent", "general"),
        sku_id=result.get("sku_id", ""),
        answer=result.get("final_response", "No response generated."),
        rag_context_used=bool(result.get("rag_context")),
    )


@router.post("/query/stream")
async def query_stream_endpoint(body: QueryRequest):
    """
    Streaming version of /query.
    Returns Server-Sent Events (SSE) — use in Streamlit with requests.get(..., stream=True).
    """
    async def event_generator():
        graph = get_graph()
        initial_state = {
            "query":          body.question,
            "intent":         "",
            "sku_id":         "",
            "rag_context":    "",
            "tool_result":    "",
            "final_response": "",
        }
        result = await asyncio.to_thread(graph.invoke, initial_state)
        response = result.get("final_response", "No response.")

        # Stream word by word
        for word in response.split(" "):
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.03)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
