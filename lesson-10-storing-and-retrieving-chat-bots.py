# Importing the os module to interact with the operating system
import os
# Importing load_dotenv function from the dotenv package to load environment variables from a .env file
from dotenv import load_dotenv

# Calling load_dotenv; this will load environment variables from a .env file into the environment
load_dotenv()
# Printing the OpenAI API key; os.environ allows you to access environment variables
print("Open AI Key:" + os.environ["OPENAI_API_KEY"])

# Importing various classes and functions for managing chat message history and conversation
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain_openai import OpenAI
from langchain.chains import ConversationChain

# Creating an instance of ChatMessageHistory to store the history of chat messages
history = ChatMessageHistory()
# Adding a user message to the history
history.add_user_message("hello! let's talk about giraffes")
# Adding an AI response to the history
history.add_ai_message("hi! i'm down to talk about giraffes")
# Converting the message history into a dictionary format
dicts = messages_to_dict(history.messages)
# Converting the dictionary format back into messages
new_messages = messages_from_dict(dicts)

# Creating an instance of the OpenAI class with a specified temperature parameter
llm = OpenAI(temperature=0)
# Creating a new ChatMessageHistory instance with the messages we previously converted from dictionary
history = ChatMessageHistory(messages=new_messages)
# Creating a ConversationBufferMemory instance, which acts as a buffer for the conversation, using the history
buffer = ConversationBufferMemory(chat_memory=history)
# Creating an instance of ConversationChain, which is used to manage the flow of the conversation
conversation = ConversationChain(llm=llm, memory=buffer, verbose=False)

# Using the conversation object to predict (generate) a response based on the input "what are they?"
print(conversation.predict(input="what are they?"))
print(conversation.predict(input="Can you repeat the last statement that you said in mandarin language with 100% accurate translation?"))

#print the memory
#print(conversation.memory)
