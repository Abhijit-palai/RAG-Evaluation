"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

    python -m evals.eval_generator
"""


from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from deepeval.models.base_model import DeepEvalBaseLLM

from src.generator import generate   # your generator: generate(query, context) -> answer
#from evals.harness import load_goldens, summarize_by_metric, print_summary

import sys
from pathlib import Path
import json
from dotenv import load_dotenv

# --- 1. RESOLVE PROJECT ROOT & SUBFOLDER PATHS ---
CURRENT_FILE = Path(__file__).resolve()
GENERATOR_EVALS_DIR = CURRENT_FILE.parent.parent  # Points to '2_Generator_evals'

# Add 2_Generator_evals so 'from src...' and 'from evals...' resolve cleanly
if str(GENERATOR_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATOR_EVALS_DIR))

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
#JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7

class HuggingFaceJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
        ).to(self.device)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=False,
            device=self.device
        )
        super().__init__(model_name)

    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        output = self.pipe(messages)
        return output[0]["generated_text"][-1]["content"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
# 2. Instantiate the custom judge
JUDGE_MODEL = HuggingFaceJudge("Qwen/Qwen2.5-3B-Instruct")


# 1. LOAD the faithfulness golden set (query + ideal_context)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

# 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
test_cases = []
for g in goldens:
    context = g["ideal_context"]              # known-good context (list of chunk strings)
    answer = generate(g["query"], context)    # RUN the generator -> actual_output

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,             # the generated answer we're judging
            retrieval_context=context,        # faithfulness checks the answer against THIS
            # no expected_output --- faithfulness never reads it
        )
    )

# 3. THE METRICS --- decompose actual_output into claims, attribute each to context
metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,   # prints WHY each score --- shows which claims were unsupported
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    ),
]

# 4. EVALUATE --- runs the metrics on every case, prints a report
result = evaluate(test_cases=test_cases, metrics=metrics)




### How to get a good Generator Evaluation
# 1) Giving a good,efficient system prompt  - > may be increase in Answer Relevance 
# 2) Also Use a good , standard LLM Model for your generator which may be varies by diff task.