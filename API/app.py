from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# initialize the flask app
app = FastAPI(
    title = 'LangChain Server',
    version = '1.0',
    description = 'An API Server'
)

# Adding routes
# 1) OpenAI route
add_routes(
    app,
    ChatOpenAI(),
    path = '/openai'
)

openai_llm = ChatOpenAI()
ollama_llm = Ollama(model='llama2')

# Prompts for our LLMs
prompt1 = ChatPromptTemplate.from_template('How many titles did {team} win from 2010 to 2023?')
prompt2 = ChatPromptTemplate.from_template('How many goals did {team} score in 2022/23 season?')

# 2) Titles prompt (prompt1) with ChatOpenAI
add_routes(
    app,
    prompt1 | openai_llm,
    path = '/titles'
)

# 3) Goals prompt (prompt2) with Ollama llm
add_routes(
    app,
    prompt2 | ollama_llm,
    path = '/goals'
    # output_type = 
)

if __name__=='__main__':
    uvicorn.run(app, host='localhost', port=8000)