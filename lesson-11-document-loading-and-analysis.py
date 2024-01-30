# Document Loading and Analysis
import os
from dotenv import load_dotenv

load_dotenv()
print("Open AI Key:" + os.environ["OPENAI_API_KEY"])

from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import chroma
from langchain.chains import RetrievalQA

loader = TextLoader(file_path="./state-of-the-union-23.txt", encoding = 'UTF-8')
documents = loader.load()

#split text into semantically related chunks as the model cannot accept the big chunk of text in one go.
#https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
store = chroma.Chroma.from_documents(texts, embeddings, collection_name="state-of-union")

llm = OpenAI(temperature=0)
chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())
print(chain.invoke({"query":"Do you provide answer to question that is out of the context of the text file?"}))
# print(chain.invoke({"query":"What did biden talk about Ohio?"}))
# print(chain.invoke({"query":"What new policies did biden talk about?"}))
# print(chain.invoke({"query":"What biden suggested so that the country will not lose out to others?"}))
# print(chain.invoke({"query":"Did Biden mention about COVID and what's the topics?"}))
# print(chain.invoke({"query":"Summarize what Biden mention in point forms."}))
