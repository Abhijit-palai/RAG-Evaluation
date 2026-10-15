# eval_rag_pipeline.py
from dotenv import load_dotenv
import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline
#from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "Generator_evals/goldens/faithfulness_dataset.json"   # reuse the queries
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


rag = RagPipeline()
# 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
with open(GOLDEN_PATH) as f:
    goldens = json.load()

# 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
test_cases = []
for g in goldens:
    result = rag.invoke(g["query"])          # retrieve -> rerank -> generate

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],       # what the generator produced
            retrieval_context=result["context"],  # what the RETRIEVER returned
        )
    )

# 3. THE THREE TRIAD METRICS
metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]

# 4. EVALUATE
result = evaluate(test_cases=test_cases, metrics=metrics)




