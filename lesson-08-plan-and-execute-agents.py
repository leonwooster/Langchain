# Plan and Execute Agents
import os

from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.chains import LLMMathChain
from langchain.agents.tools import Tool

load_dotenv();
print("Open AI Key:" + os.environ["OPENAI_API_KEY"])
print("SerpAPI Key:" + os.environ["SERPAPI_API_KEY"])

search = SerpAPIWrapper() #need to "pip install google-search-results"
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
wikipedia = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",        
    ),
    Tool(
        name="Wikipedia",
        func = wikipedia.run,
        description = "useful for when you need to look up facts and statistics"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]

model = ChatOpenAI( #There's error of token exceeding 4097 for chat-gpt-3.5-turbo
    model="gpt-4",    
    temperature=0
    )
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

prompt = "Where will the next summer olympics be hosted? What is the population of that country raised to the 0.43 power?"

agent.invoke({"input": prompt})

