import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# create Session state in streamlit
# Session state remembers the stuff we put in it, across sessions
if 'vector' not in st.session_state:
    st.session_state['embeddings'] = OllamaEmbeddings()
    st.session_state['web_content'] = WebBaseLoader('https://www.datascienceportfol.io/nvkanirudh').load()
    st.session_state['chunked_web_content'] = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(st.session_state['web_content'])
    st.session_state['db'] = FAISS.from_documents(st.session_state['chunked_web_content'], st.session_state['embeddings'])

st.title('Chatting With LLMs Using Groq')

# initializing the ChatGroq along with appropriate open source llm
llm = ChatGroq(groq_api_key=groq_api_key, model='Gemma-7b-It')

# initializing the prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context. 
    context: {context},
    question: {input}
    """
)

# Initializing chain and retrievers for retrieval_chain
chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state['db'].as_retriever()
retrieval_chain = create_retrieval_chain(retriever, chain)

input = st.text_input('Please enter your prompt:')

if input:
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': input})
    print("Response Time:", time.process_time() - start_time)
    st.write(response['answer'])

    # streamlit expander
    with st.expander('Document Similarity Search'):
        # Finding the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-----------------------------------')
