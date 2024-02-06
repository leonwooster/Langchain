'''Demonstrating how to chain a prompt with a LLM model'''
# Prompts and LLMs
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()
print("Key:" + os.environ["OPENAI_API_KEY"])

template = "You are a naming consultant for new companies. What is a good name for a {company} that makes {product}?"

prompt = PromptTemplate.from_template(template)
llm = OpenAI(temperature=0.9)

chain = LLMChain(llm=llm, prompt=prompt)

#A dictionary of key value pairs.
print(chain.invoke({"company": "ABC Startup", "product": "colorful socks"}))
