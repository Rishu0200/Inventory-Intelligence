"""
POST /api/query — Natural language query endpoint.
Runs the LangGraph orchestrator and returns the final response.
"""
from __future__ import annotations
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import QueryRequest, QueryResponse
from orchestrator.graph import get_graph

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest):
    """
    Ask any inventory question in natural language.
    """
    initial_state = {
        "query":          body.question,
        "intent":         "",
        "sku_id":         "",
        "rag_context":    "",
        "tool_result":    "",
        "final_response": "",
    }

    try:
        graph  = get_graph()
        result = graph.invoke(initial_state)
    except Exception as e:
        return QueryResponse(
            question=body.question,
            intent="error",
            sku_id="",
            answer=f"Sorry, something went wrong processing that query. ({e})",
            rag_context_used=False,
        )

    rag_context = result.get("rag_context", "")
    rag_used = bool(rag_context) and rag_context.strip() != "No relevant documents found."

    return QueryResponse(
        question=body.question,
        intent=result.get("intent", "general"),
        sku_id=result.get("sku_id", ""),
        answer=result.get("final_response", "No response generated."),
        rag_context_used=rag_used,
    )


async def query_stream_endpoint(body: QueryRequest):
    async def event_generator():
        initial_state = {
            "query":          body.question,
            "intent":         "",
            "sku_id":         "",
            "rag_context":    "",
            "tool_result":    "",
            "final_response": "",
        }
        try:
            graph    = get_graph()
            result   = await asyncio.to_thread(graph.invoke, initial_state)
            response = result.get("final_response", "No response.")
        except Exception as e:
            response = f"Sorry, something went wrong processing that query. ({e})"

        for word in response.split(" "):
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.03)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
