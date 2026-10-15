# Normal generator code for rag. 
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint , HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# model = HuggingFaceEndpoint(
#     model = 'meta-llama/Llama-3.2-3B-Instruct',
#     task = 'text_generation'
# )
# llm = ChatHuggingFace(llm = model , temperature = 0)

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    max_new_tokens=256,
    temperature=0,
    do_sample=False,
    #device_map="auto"
)

llm = HuggingFacePipeline(pipeline=pipe)

#IMP:- write this prompt as good as for better result in evaluation
prompt = ChatPromptTemplate.from_template(
    """You are a helpful teaching assistant for a course on LLM evaluations.
Answer the student's question using ONLY the context provided below.

Rules:
- Use only information present in the context. Do not add outside knowledge.
- If the context does not contain enough information to answer, say:
  "I don't have enough information in the course material to answer that."
- Keep the answer clear and concise.

Context:
{context}

Question: {question}

Answer:"""
)

parser = StrOutputParser()

chain = prompt | llm | parser

def generate(query:str , context:list[str])-> str:
    """Generate a grounded answer from the query and context chunks."""
    
    context_text = "\n\n".join(context)
    
    return chain.invoke({'question':query , "context":context_text})


# A quick manual test 

# if __name__ =='__main__':
#     ctx = [
#         "Online evaluation in LLM applications means evaluating the system using real user interactions while the application is running in production."
#         "In simple words, it checks how well your LLM or RAG application performs when actual users ask questions."
#     ]
#     print(generate("What is the online eval ??" , ctx))