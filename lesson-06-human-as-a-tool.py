# Simple Sequential Chains
import os
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

load_dotenv()
print("Key:" + os.environ["OPENAI_API_KEY"])

llm = ChatOpenAI(temperature=0.0)
math_llm = OpenAI(temperature=0.0)
tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_chain.invoke({"input":"What's my friend Andi's surname?"})
