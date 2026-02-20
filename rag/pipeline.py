from typing import List
from chunking.simple_chunker import chunk_text
from embeddings.embedder import embed_chunks
from vector_store.in_memory_store import InMemoryVectorStore
from retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self, documents: List[str], chunk_size: int = 10):
        """
        Initialize RAG pipeline with documents.

        Pipeline: docs → chunks → embeddings → vector store → retriever
        """
        # Step 1: Chunk all documents
        self.all_chunks = []
        for doc in documents:
            chunks = chunk_text(doc, chunk_size)
            self.all_chunks.extend(chunks)

        # Step 2: Embed chunks
        self.vectors = embed_chunks(self.all_chunks)

        # Step 3: Build vector store
        self.store = InMemoryVectorStore()
        for chunk, vector in zip(self.all_chunks, self.vectors):
            self.store.add(chunk, vector)

        # Step 4: Create retriever
        self.retriever = Retriever(self.store)

        print(
            f"Initialized RAG pipeline: {len(self.all_chunks)} chunks indexed")

    def query(self, question: str, top_k: int = 2) -> List[str]:
        """
        Retrieve relevant context chunks for a question.
        """
        return self.retriever.retrieve(question, top_k=top_k)
