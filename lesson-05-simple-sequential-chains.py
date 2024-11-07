'''Demonstrate how different steps can be chained together and executed one after another one using 
the same AI model'''
# Simple Sequential Chains
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

# Load environment variables from .env file
load_dotenv()
print("Key:" + os.environ["OPENAI_API_KEY"])

llm = OpenAI(temperature=0)
template = "What is a good name for a company that makes {product}?"
first_prompt = PromptTemplate.from_template(template)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

second_template = "Write a catch phrase for the following company: {company_name}"
second_prompt = PromptTemplate.from_template(second_template)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

'''output from first chain becomes input of second chain'''
overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)
catchphrase = overall_chain.invoke(input= {"input": "variety of Nyonya kueh"})
print(catchphrase)
