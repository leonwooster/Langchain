# Using different models with LangChain
import os
from dotenv import load_dotenv

# Load environment variables from .env file
#load_dotenv()
#print("Key:" + os.environ["OPENAI_API_KEY"])

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

repo_id = "google/flan-t5-xxl"

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))


# from langchain import HuggingFaceHub

# llm = HuggingFaceHub(
#     repo_id="google/flan-t5-base", model_kwargs={"temperature": 0, "max_length": 64}
# )
# prompt = "What are good fitness tips?"

# print(llm(prompt))
