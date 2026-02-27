User Request
     │
     ▼
FastAPI Endpoint
     │
     ▼
Cache Layer
     │
     ├── Cache Hit → Response
     │
     ▼
RAG Pipeline
     │
     ├─ Chunking
     ├─ Embedding
     ├─ Vector Store
     └─ Retriever
     │
     ▼
Context Response
