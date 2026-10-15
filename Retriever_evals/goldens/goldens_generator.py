# Golden dataset -> questions with ideal answers creation by deep evals.
# It will be not so good working mechanism but just try it out.

import os, re, glob, json, random
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM
import time
load_dotenv()


# 1. Groq Custom Judge / Generator wrapper for DeepEval
class GroqSynthesizerModel(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model_name = model_name
        self.model = ChatGroq(model=model_name, temperature=0.3)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        # Retry with exponential backoff on Groq rate limits
        max_retries = 5
        base_delay = 3
        for attempt in range(max_retries):
            try:
                # Add a brief pause between consecutive calls to pace requests
                time.sleep(1.0)
                return self.model.invoke(prompt).content
            except Exception as e:
                # Check for rate limit error (429)
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = base_delay * (attempt + 1)
                    print(f"\n[!] Groq TPM hit (429). Backing off for {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise RuntimeError("Exceeded maximum retries due to Groq rate limits.")

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# --- reuse your own VTT cleaning + chunking (same as the retriever) ---
def load_chunks():
    texts = []
    for path in glob.glob("Data/*.vtt"):
        with open(path) as f:
            lines = [ln.strip() for ln in f
                     if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln]
        texts.append(" ".join(lines))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))


# --- generate ---
chunks = load_chunks()
if not chunks:
    raise SystemExit("Exiting: No chunks available to synthesize goldens.")

sample = random.sample(chunks, min(15, len(chunks)))
contexts = [[c] for c in sample]

# Initialize Groq LLM as the synthesizer model
groq_llm = GroqSynthesizerModel(model_name="openai/gpt-oss-120b")

synthesizer = Synthesizer(
    model=groq_llm,
    async_mode=False  # Keep False to respect Groq free-tier RPM limits
)

print(f"Generating synthetic goldens from {len(contexts)} chunks using Groq...")
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1,
)


# --- convert to YOUR schema (id / query / ideal_answer / source) ---
rows = []
for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",        # Synthesizer won't know the session -- you fill this
    })

with open("goldens/retriever_deepeval_goldens.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
    
print(f"wrote {len(rows)} DRAFT goldens -> goldens/component_goldens_draft.json")
print("!! REVIEW EVERY ONE before using: check grounding, trim padding, fix leading questions.")