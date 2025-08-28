import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_community.vectorstores import FAISS

# For genai embeddings
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

import asyncio # NEW: Import asyncio

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY   

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    genai.configure(api_key=GOOGLE_API_KEY)

    # Define an async function to handle embedding creation and the embedding model initialization
    async def create_embeddings_and_model():
        # Instantiate the LangChain embedding model for query embedding
        query_embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        chunk_embeddings = []
        for chunk in text_chunks:
            response = genai.embed_content( # genai.embed_content is synchronous, no need for await here unless using an async version
                model="models/embedding-001",
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT"
            )
            chunk_embeddings.append(response['embedding'])
        return query_embeddings_model, chunk_embeddings

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError: 
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the async function
    query_embeddings_model, chunk_embeddings = loop.run_until_complete(create_embeddings_and_model())

    # Create the FAISS vector store
    vector_store = FAISS.from_embeddings(
        text_embeddings=zip(text_chunks, chunk_embeddings),
        embedding=query_embeddings_model 
    )
    return vector_store


def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) 
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vector_store.as_retriever(),memory=memory )
    return conversation_chain