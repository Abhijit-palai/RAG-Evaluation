# eval_retriever.py
from dotenv import load_dotenv
import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
import time

# from src.reranker import RerankingRetriever
# from evals.harness import load_goldens, summarize_by_metric, print_summary
from src.retriever import build_retriever

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
#JUDGE_MODEL = "gpt-4o-mini"   # NOTE: differs from the other evals (gpt-4o-mini)
THRESHOLD = 0.7


# Custom Groq Judge for DeepEval don't need to give importance.
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model_name = model_name
        self.model = ChatGroq(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        max_retries = 5
        base_delay = 3
        for attempt in range(max_retries):
            try:
                time.sleep(1.0)  # Pacing to protect TPM limit
                return self.model.invoke(prompt).content
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = base_delay * (attempt + 1)
                    print(f"\n[!] Groq 429 rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise RuntimeError("Exceeded retry limit on Groq rate limits.")

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# Instantiate the judge model
JUDGE_MODEL = GroqJudge(model_name="openai/gpt-oss-120b")  # we write only this line if we have a subsribed api key like openai , claude etc.
# This above func is only for groq free tier model initialization.


# 1. Load the golden set ----- the fixed, human-authored truth

with open (GOLDEN_PATH , "r", encoding="utf-8") as f:
    goldens = json.load(f)

# 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
#    then build one test case per golden.

retriever = build_retriever()

test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g['query'])
    retrieval_context = [doc.page_content for doc in retrieved]
    
    test_cases.append(
        LLMTestCase(
            input=g['query'],
            expected_output = g['ideal_answer'],
            retrieval_context= retrieval_context,
            actual_output = "(generator not evaluated in this run)",
        )
    )
    
# 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)

metrics = [
    ContextualRecallMetric(threshold=THRESHOLD , model = JUDGE_MODEL , include_reason=True),
    ContextualPrecisionMetric(threshold = THRESHOLD , model = JUDGE_MODEL , include_reason = True),
]

#4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "base_k5",          # vs "reranked" when you swap it in
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 3,
        "judge_model": JUDGE_MODEL.get_model_name(),
        "golden_set": GOLDEN_PATH,
    },
)

### 1) At first run Baseline :-

# precision = 80 
# recall = 80

# pass = 10 out of 15
# fail = 5

### 2) second Baseline Eval -> after use reranker and chunk_size = 1000 and chunk_overlap = 150 increase  
# precision = 85  -------> by use of reranker
# recall = 92   -----> by use of more chunk_size and chunk_overlap

# pass = 13 out of 15
# fail = 2

### 3) Third Baseline Eval -> after evaluate embedding model 
# precision = 85  
# recall = 99   

# pass = 12 out of 15
# fail = 3

# How to evaluate ??
# -> increase or decrease chunk_size and chunk overlap for retriever.
# -> use reranker for valuable rank according to 'K' size for increasing contextual precision.
# -> Evaluate Embedding model for retriever.
# -> try a standard 'K' value.