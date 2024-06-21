import streamlit as st
import re
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

# Function to wrap text while preserving newlines
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = "\n".join(wrapped_lines)
    return wrapped_text

# Function to clean up the response
def clean_response(response):
    # Remove specific unwanted instruction from the response
    instruction = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
    response = response.replace(instruction, "").strip()
    
    # Remove text within parentheses or brackets
    response = re.sub(r'\(.*?\)', '', response)
    response = re.sub(r'\[.*?\]', '', response)
    
    # Further clean up response by removing any remaining non-essential text
    response = re.sub(r'^\s*Answer:\s*', '', response)
    
    return response.strip()

# Load the document
@st.cache_data
def load_document():
    loader = TextLoader("MG University.txt")
    document = loader.load()
    return document

# Split the text into chunks
@st.cache_data
def split_document(_document):
    text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
    docs = text_splitter.split_documents(_document)
    return docs

# Create embeddings and vector store
@st.cache_data
def create_vector_store(_docs):
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(_docs, embeddings)
    return db

# Set up the HuggingFaceHub LLM
@st.cache_data
def setup_llm():
    llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token="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", model_kwargs={"temperature":.3,"max_length":100})
    llm.client.api_url = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'
    return llm

# Load QA chain
@st.cache_data
def load_qa_chain_model(_llm):
    chain = load_qa_chain(_llm, chain_type="stuff")
    return chain

# Initialize Streamlit app
st.title("School Of Data Analytics")
st.write("Ask questions about School Of Data Analytics")

# Load resources
document = load_document()
docs = split_document(document)
db = create_vector_store(docs)
llm = setup_llm()
chain = load_qa_chain_model(llm)

# Maintain conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Display conversation history
for i, (speaker, text) in enumerate(st.session_state.history):
    if speaker == "User":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")

# User input
user_input = st.text_area("You: ", height=100)


if st.button("Send"):
    if user_input:
        # Perform similarity search
        docs_result = db.similarity_search(user_input)
        
        if docs_result:
            # Get the answer using QA chain
            answer = chain.run(input_documents=docs_result, question=user_input)
            cleaned_answer = clean_response(answer)
            st.session_state.history.append(("User", user_input))
            st.session_state.history.append(("Bot", cleaned_answer))
        else:
            st.session_state.history.append(("User", user_input))
            st.session_state.history.append(("Bot", "No relevant information found."))

    # Clear the input box after sending
    st.experimental_rerun()
