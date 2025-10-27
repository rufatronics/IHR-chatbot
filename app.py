# app.py

import os
import streamlit as st
from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIGURATION ---
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
VECTOR_DB_PATH = "./chroma_db"
RAG_DATASET = "sdiazlor/rag-human-rights-from-files"
RAG_PROMPT_TEMPLATE = """
You are an expert AI assistant specializing in International Law, specifically focusing on Human Rights.
Answer the user's question ONLY based on the context provided below.
If the context does not contain the answer, politely state that you cannot answer based on the provided documents.

CONTEXT:
---
{context}
---

QUESTION: {input}
"""
prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

@st.cache_resource
def setup_rag_system():
    """Initializes and caches the RAG system components."""
    try:
        # Load API Key from Streamlit Secrets
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("Google API Key not found in Streamlit secrets. Please configure it.")
            return None, None
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        
        # 1. Document Loading and Chunking
        st.info(f"Loading and processing documents from Hugging Face dataset: {RAG_DATASET}...")
        dataset = load_dataset(RAG_DATASET, split="train")
        # Use a subset of text data (e.g., the 'text' column)
        texts = dataset['text'][:50] # Use first 50 docs for a fast demo
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )
        # Create LangChain Documents (we don't preserve metadata for simplicity here)
        documents = text_splitter.create_documents(texts)
        
        # 2. Embedding and Ingestion
        embedding_function = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Ingest documents into ChromaDB
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embedding_function,
            persist_directory=VECTOR_DB_PATH
        )
        
        # 3. Create RAG Chain
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        return rag_chain, len(documents)
        
    except Exception as e:
        st.error(f"Error during RAG setup: {e}")
        return None, None

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="International Law RAG Assistant ⚖️")
st.title("International Law RAG Assistant ⚖️")
st.markdown("Ask questions about **Human Rights** based on a specialized legal document dataset.")

# Setup the RAG system (cached to run once)
rag_chain, num_chunks = setup_rag_system()

if rag_chain:
    st.sidebar.success(f"RAG System Ready! ({num_chunks} chunks indexed)")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt_text := st.chat_input("Ask a question about International Human Rights..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                try:
                    # Invoke the RAG chain
                    response = rag_chain.invoke({"input": prompt_text})
                    
                    answer = response["answer"]
                    sources = response["context"]
                    
                    # Display the answer
                    st.markdown(answer)

                    # Display the context used in an expander
                    with st.expander("🔍 Context Used"):
                        for i, doc in enumerate(sources):
                            st.text_area(f"Source Document {i+1}", doc.page_content, height=100, disabled=True)
                            
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})

else:
    st.error("RAG system failed to initialize. Check your API key and network connection.")

