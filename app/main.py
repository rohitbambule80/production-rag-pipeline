from fastapi import FastAPI

app = FastAPI(
    title="Production RAG Pipeline",
    description="End-to-end RAG system with chunking, embeddings, and vector search",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "ok",
        "service": "rag-pipeline",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
