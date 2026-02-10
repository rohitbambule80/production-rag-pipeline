from fastapi import APIRouter, HTTPException
from app.schemas import RAGQueryRequest, RAGQueryResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest):
    """
    Retrieve relevant context chunks for a RAG query using vector similarity search.

    - Embeds query → searches vector store → returns top_k chunks
    - Context ready for LLM augmentation
    """
    # Mock response (RAG logic wiring next step)
    mock_context = [
        "RAG reduces hallucinations by retrieving relevant documents from a vector store.",
        "Documents are chunked, embedded, and stored for fast similarity search."
    ]

    response = RAGQueryResponse(
        query=request.query,
        context=mock_context[:request.top_k],  # Respect top_k limit
        context_count=len(mock_context[:request.top_k]),
        metadata={
            "model": "toy-embedding-v1",
            "chunk_size": 10,
            "retriever": "cosine-similarity"
        }
    )

    return response
