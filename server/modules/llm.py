from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a professional AI assistant. Your job is to provide clear, accurate, and helpful responses 
based **only on the information provided in the context** below.

---

🔍 **Context**:
{context}

🙋‍♂️ **User Question**:
{question}

---

💬 **Answer**:
- Respond in a professional, calm, and respectful tone.
- Base your answer **only on the provided context**.
- If the context does not contain the information needed, politely say: 
  "I'm sorry, but I don't have enough information to answer that based on the provided context."
- Do NOT make up facts or provide information outside of the context.
- Use simple and clear explanations where possible.
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )