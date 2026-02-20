from fastapi import FastAPI
from app.api import router as rag_router, get_pipeline
from app.schemas import HealthResponse

app = FastAPI(
    title="Production RAG Pipeline",
    description="End-to-end RAG system: chunk → embed → store → retrieve → API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register API routes
app.include_router(rag_router)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Production health check for monitoring/load balancers.
    Accesses pipeline safely via service accessor.
    """
    pipeline = get_pipeline()

    return {
        "status": "healthy",
        "service": "rag-pipeline",
        "version": "1.0.0",
        "documents_indexed": len(pipeline.all_chunks)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
