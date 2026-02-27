from typing import List, Dict
from rag.pipeline import RAGPipeline


class RAGEvaluator:
    """
    Simple evaluation framework for RAG retrieval quality.
    """

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def evaluate(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate retrieval accuracy.

        test_cases format:
        [
            {"query": "...", "expected_keyword": "..."},
            ...
        ]
        """

        total = len(test_cases)
        correct = 0
        results = []

        for case in test_cases:
            query = case["query"]
            keyword = case["expected_keyword"]

            retrieved = self.pipeline.query(query)

            hit = any(keyword.lower() in chunk.lower() for chunk in retrieved)

            if hit:
                correct += 1

            results.append({
                "query": query,
                "expected_keyword": keyword,
                "retrieved": retrieved,
                "success": hit
            })

        accuracy = correct / total if total else 0

        return {
            "accuracy": round(accuracy, 3),
            "total_tests": total,
            "passed": correct,
            "results": results
        }


if __name__ == "__main__":

    documents = [
        "Large Language Models hallucinate when they lack grounding.",
        "Retrieval Augmented Generation reduces hallucinations using document retrieval.",
        "Vector databases enable similarity search using embeddings."
    ]

    pipeline = RAGPipeline(documents)

    evaluator = RAGEvaluator(pipeline)

    test_cases = [
        {
            "query": "How does RAG reduce hallucinations?",
            "expected_keyword": "hallucinations"
        },
        {
            "query": "What enables vector similarity search?",
            "expected_keyword": "vector"
        }
    ]

    report = evaluator.evaluate(test_cases)

    print("\nRAG Evaluation Report")
    print("---------------------")
    print(report)
