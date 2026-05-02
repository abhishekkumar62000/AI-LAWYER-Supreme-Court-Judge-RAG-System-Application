from groq import Groq, BadRequestError
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Uncomment the following if you're NOT using pipenv
#from dotenv import load_dotenv
#load_dotenv()

#Step1: Setup LLM (Use DeepSeek R1 with Groq)
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Please set it in environment or .env file.")
groq_client = Groq(api_key=groq_api_key)

def resolve_groq_model(client: Groq) -> str:
    env_model = os.environ.get("GROQ_MODEL")
    if env_model:
        return env_model
    try:
        models = client.models.list()
        available_ids = [m.id for m in getattr(models, "data", [])]
        preferred = os.environ.get(
            "GROQ_MODEL_PREFERENCE",
            "llama-3.3-70b-versatile, llama-3.2-11b-text-preview, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it",
        )
        prefs = [m.strip() for m in preferred.split(",") if m.strip()]
        for p in prefs:
            if p in available_ids:
                return p
        for mid in available_ids:
            if "llama" in mid:
                return mid
        if available_ids:
            return available_ids[0]
    except Exception:
        pass
    return "llama-3.2-11b-text-preview"

GROQ_MODEL = resolve_groq_model(groq_client)

#Step2: Retrieve Docs

def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

#Step3: Answer Question

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    final_prompt = prompt.format(question=query, context=context)
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except BadRequestError as e:
        msg = getattr(e, "message", str(e))
        if "model" in msg and ("decommissioned" in msg or "not found" in msg):
            fallback_model = resolve_groq_model(groq_client)
            completion = groq_client.chat.completions.create(
                model=fallback_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,
            )
            return completion.choices[0].message.content
        raise

#question="If a government forbids the right to assemble peacefully which articles are violated and why?"
#retrieved_docs=retrieve_docs(question)
#print("AI Lawyer: ",answer_query(documents=retrieved_docs, model=llm_model, query=question))