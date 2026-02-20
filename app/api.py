from fastapi import APIRouter
from app.schemas import RAGQueryRequest, RAGQueryResponse
from rag.pipeline import RAGPipeline

router = APIRouter(prefix="/rag", tags=["rag"])

# Production: Load from database/S3/filesystem
DOCUMENTS = [
    "Large Language Models hallucinate when they lack grounding in external knowledge.",
    "Retrieval Augmented Generation (RAG) reduces hallucinations by retrieving and injecting relevant documents into LLM prompts.",
    "Vector databases like Pinecone enable fast similarity search over text embeddings using algorithms like HNSW.",
    "RAG pipelines typically chunk documents, embed chunks, store in vector DB, and retrieve top-k matches for queries."
]

# Initialize once at module load (singleton pattern)
rag_pipeline = RAGPipeline(DOCUMENTS)


@router.post("/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest):
    """
    Retrieve relevant context chunks for a RAG query using vector similarity search.

    Embeds incoming query → ANN search → returns top_k most relevant chunks.
    Context ready for downstream LLM augmentation.
    """
    context_chunks = rag_pipeline.query(
        question=request.query,
        top_k=request.top_k
    )

    return RAGQueryResponse(
        query=request.query,
        context=context_chunks,
        context_count=len(context_chunks),
        metadata={
            "pipeline": "production-rag-v1",
            "retriever": "cosine-similarity",
            "chunk_count": len(rag_pipeline.all_chunks),
            "model": "toy-embedding-26d"
        }
    )
