import ollama
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Constants
PDF_FILE_PATH = './data/2023-Annual-Report-1.pdf'  # Update with your file path
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
OLLAMA_MODEL_NAME = 'mistral'

# Function to load and split the PDF document
def load_and_split_pdf(file_path):
    """
    Loads a PDF document and splits it into pages.
    Args:
    file_path (str): Path to the PDF file.
    Returns:
    list: List of pages from the PDF document.
    """
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# Function to split text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    """
    Splits text from pages into smaller chunks.
    Args:
    pages (list): List of pages from the document.
    chunk_size (int): Size of each text chunk.
    chunk_overlap (int): Overlap between consecutive chunks.
    Returns:
    list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

# Function to create embeddings and vector store
def create_embeddings_and_vectorstore(splits, model_name):
    """
    Creates embeddings and a vector store from document splits.
    Args:
    splits (list): List of document splits.
    model_name (str): Name of the Ollama model to use.
    Returns:
    Chroma: Vector store object.
    """
    embeddings = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")
    return Chroma.from_documents(documents=splits, embedding=embeddings)

# Function to format documents for context
def format_docs(docs):
    """
    Formats a list of documents into a single string.
    Args:
    docs (list): List of documents.
    Returns:
    str: Formatted string of document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# Function to query Ollama LLM with context
def ollama_llm(question, context):
    """
    Queries the Ollama LLM with a question and context.
    Args:
    question (str): The question to ask.
    context (str): The context for the question.
    Returns:
    str: The response from the LLM.
    """
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Function to execute the RAG chain
def rag_chain(question, retriever):
    """
    Executes the RAG chain to get an answer to a question.
    Args:
    question (str): The question to ask.
    retriever: The retriever object for document retrieval.
    Returns:
    str: The answer to the question.
    """
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Main Execution Flow
def main():
    """
    Main function to execute the RAG workflow.
    """
    pages = load_and_split_pdf(PDF_FILE_PATH)
    splits = split_text(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    vectorstore = create_embeddings_and_vectorstore(splits, OLLAMA_MODEL_NAME)
    retriever = vectorstore.as_retriever()

    # Example Queries
    query1 = "How does Nvidia define its fiscal year, and what were the lengths of fiscal years 2024 and 2023?"
    result1 = rag_chain(query1, retriever)
    print("Result 1:", result1)

    query2 = "What were the key highlights and financial performance of Nvidia's Data Center segment in the third quarter of fiscal year 2024?"
    result2 = rag_chain(query2, retriever)
    print("Result 2:", result2)

@cl.on_chat_start
def on_chat_start():
    print("A new chat session has started!")
    main()

@cl.on_message
def on_message(msg: cl.Message):
    print("The user sent: ", msg.content)
            
#if __name__ == "__main__":
    #main()