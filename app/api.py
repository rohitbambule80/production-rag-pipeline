from fastapi import APIRouter
from app.schemas import RAGQueryRequest, RAGQueryResponse
from rag.pipeline import RAGPipeline
from rag.cache import SimpleCache

router = APIRouter(prefix="/rag", tags=["rag"])

DOCUMENTS = [
    "Large Language Models hallucinate when they lack grounding in external knowledge.",
    "Retrieval Augmented Generation (RAG) reduces hallucinations by retrieving and injecting relevant documents into LLM prompts.",
    "Vector databases like Pinecone enable fast similarity search over text embeddings using algorithms like HNSW.",
    "RAG pipelines typically chunk documents, embed chunks, store in vector DB, and retrieve top-k matches for queries."
]

rag_pipeline = RAGPipeline(DOCUMENTS)
cache = SimpleCache()


def get_pipeline():
    return rag_pipeline


@router.post("/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest):

    # 🔹 1. Try cache first
    cached = cache.get(request.query)
    if cached:
        return RAGQueryResponse(
            query=request.query,
            context=cached,
            context_count=len(cached),
            metadata={
                "pipeline": "production-rag-v1",
                "cache": "hit"
            }
        )

    # 🔹 2. Run retrieval
    context_chunks = rag_pipeline.query(
        question=request.query,
        top_k=request.top_k
    )

    # 🔹 3. Store in cache
    cache.set(request.query, context_chunks)

    return RAGQueryResponse(
        query=request.query,
        context=context_chunks,
        context_count=len(context_chunks),
        metadata={
            "pipeline": "production-rag-v1",
            "cache": "miss",
            "chunk_count": len(rag_pipeline.all_chunks),
            "model": "toy-embedding-26d"
        }
    )
