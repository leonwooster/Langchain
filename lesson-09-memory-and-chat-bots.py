# Memory and Chat Bots
import os
from dotenv import load_dotenv

load_dotenv();
print("Open AI Key:" + os.environ["OPENAI_API_KEY"])

from langchain_openai import OpenAI
from langchain.chains import ConversationChain

# Printing Predictions
llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=False)

#send the message to the AI and expect a respond from it.
print(conversation.predict(input="Hi there!")) 
print(conversation.predict(input="Can we talk about weather?"))
print(conversation.predict(input="It's a beautiful day today"))

# -------------------------------------------------------------
# Creating an interactive terminal Chat Bot
llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm)

print("Welcome to your AI chatbot! What's on your mind?")
for _ in range(0, 3):
    human_input = input("You: ")
    ai_response = conversation.predict(input=human_input)
    print(f"AI: {ai_response}")
