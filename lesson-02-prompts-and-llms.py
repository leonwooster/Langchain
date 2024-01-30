# Prompts and LLMs
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
print("Key:" + os.environ["OPENAI_API_KEY"])
      
#get from Windows environment variable.
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#print("Key:" + os.environ["OPENAI_API_KEY"])
#os.environ["OPENAI_API_KEY"] = ""

client = OpenAI()
prompt = "What would a good company name be for a company that makes colorful socks?"

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    stream=False,
)

for respond in completion.choices:
    print(respond.message.content)

#for streaming
# for chunk in stream:
#     print(chunk.choices[0].delta.content or "", end="")

