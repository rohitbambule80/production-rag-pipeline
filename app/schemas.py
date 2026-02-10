from pydantic import BaseModel, Field
from typing import List


class RAGQueryRequest(BaseModel):
    """Request schema for RAG queries."""
    query: str = Field(..., min_length=1, max_length=500,
                       description="User question")
    top_k: int = Field(
        2, ge=1, le=10, description="Number of context chunks to retrieve")


class RAGQueryResponse(BaseModel):
    """Response schema for RAG retrieval."""
    query: str = Field(..., description="Echo of original query")
    context: List[str] = Field(...,
                               description="Retrieved relevant document chunks")
    context_count: int = Field(..., description="Number of chunks returned")
    metadata: dict = Field({"model": "toy-embedding-v1"},
                           description="Pipeline metadata")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    service: str
    version: str = "1.0.0"
