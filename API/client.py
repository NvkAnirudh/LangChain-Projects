import requests
import streamlit as st

# Function to get OpenAI response
def get_openai_response(input_text):
    response = requests.post('http://localhost:8000/titles/invoke', json={'input':{'team':input_text}})
    print(response.json())
    return response.json()['output']['content']

# Function to get Ollama LLM's response
def get_ollama_response(input_text):
    response = requests.post('http://localhost:8000/goals/invoke', json={'input':{'team':input_text}})
    print(response.json())

    return response.json()['output']

# Streamlit web app
st.title('Football Info')
input_text_titles = st.text_input('No. of titles won')
input_text_goals = st.text_input('No. of goals won')

if input_text_titles:
    st.write(get_openai_response(input_text_titles))

if input_text_goals:
    st.write(get_ollama_response(input_text_goals))

