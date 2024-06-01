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
from nemoguardrails import LLMRails, RailsConfig

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Constants ---
DEFAULT_PDF_PATH = "data/CrashCourse_Info_Cohort4.pdf"  # Update with your PDF path
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

# Configuring the guardrails
yaml_content = """
rules:
  - name: safe_interaction
    description: |
      Ensure that the assistant does not provide harmful or inappropriate responses.
    triggers:
      - input: "What do you think about the *?"
    actions:
      - respond: "I don't have personal opinions, but I'm here to help with your questions about the Build Fast With AI Course."

  - name: context_awareness
    description: |
      Ensure the assistant provides answers based on the provided context.
    triggers:
      - input: "Tell me about *"
    actions:
      - context-aware-response

  - name: clarify_missing_information
    description: |
      Ensure the assistant asks for clarification when the question is unclear or lacks information.
    triggers:
      - input: "Explain *"
    actions:
      - respond: "Could you please provide more details about what you want to know?"

responses:
  safe_interaction:
    - "I don't have personal opinions, but I'm here to help with your questions about the Build Fast With AI Course."
  context_awareness:
    - "Based on the provided context, here is what I can tell you..."
  clarify_missing_information:
    - "Could you please provide more details about what you want to know?"

settings:
  max_tokens: 512
  temperature: 0.3
"""

colang_content = """
define context-aware-response:
  If the input contains context-specific information:
    Use the context to generate a detailed response.
  Otherwise:
    Ask the user for more context.

on safe_interaction:
  Trigger: If the input asks for personal opinions.
  Action: Respond with a predefined message to avoid personal opinions.

on context_awareness:
  Trigger: If the input requests information about a specific topic.
  Action: Use the provided context to generate a detailed response or ask for more context.

on clarify_missing_information:
  Trigger: If the input is vague or lacks details.
  Action: Ask the user for more information to clarify their request.
"""

config = RailsConfig.from_content(
    yaml_content=yaml_content,
    colang_content=colang_content
)
rails = LLMRails(config)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "I can't answer this question, happy to discuss with Build Fast With AI Course Chatbot". 
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
            # Apply NeMo Guardrails
            bot_message = rails.generate(messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": response["output_text"]}])
            # Display assistant's response
            st.write(bot_message['content'])
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_message['content']})
