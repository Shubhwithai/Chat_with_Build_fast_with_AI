import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Constants ---
DEFAULT_PDF_PATH = "/data/CrashCourse_Info_Cohort4.pdf"  # Update with your PDF path
VECTOR_STORE_FILENAME = "faiss_index"

# --- Initialize Vector Store ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = None

# --- Functions ---
def get_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_and_save_vector_store(text_chunks):
    global vector_store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_STORE_FILENAME)

def load_vector_store():
    global vector_store
    vector_store = FAISS.load_local(VECTOR_STORE_FILENAME, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "I cant answer thais Question, happy to discuss with this Crash Course..". 
    Do not provide incorrect answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- Streamlit UI ---
st.set_page_config(page_title="Course Chatbot", page_icon="📃💬")
st.title("💬 Chat with Build Fast With AI ")

# --- Session State Initialization ---
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about the Course! 😊"}
    ]

# --- PDF Processing & Vector Store ---
if not os.path.exists(VECTOR_STORE_FILENAME):
    with st.spinner("Processing PDF..."):
        raw_text = get_pdf_text(DEFAULT_PDF_PATH)
        text_chunks = get_text_chunks(raw_text)
        create_and_save_vector_store(text_chunks)
        st.success("PDF processed and vector store created!")

if os.path.exists(VECTOR_STORE_FILENAME) and vector_store is None:
    with st.spinner("Loading vector store..."):
        load_vector_store()
        # st.success("Vector store loaded!")

# --- Chat Interaction ---
# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input area
if prompt := st.chat_input("Ask your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message first
    with st.chat_message("user"):
        st.write(prompt)

    # Process user input
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
                docs = vector_store.similarity_search(prompt)
                chain = get_conversational_chain()
                response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
                # Display assistant's response
                st.write(response["output_text"])
                # Add assistant's response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
