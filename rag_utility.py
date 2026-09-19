import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables from .env file
load_dotenv()

# Get the current working directory
working_dir = os.path.dirname(os.path.abspath(__file__))


# Load the embedding model
embedding = HuggingFaceEmbeddings()


# Load the Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# Document ingestion process

# def document_process_to_Chroma_db(file_name):

#     # Load the document
#     loader = UnstructuredFileLoader(
#         f"{working_dir}/{file_name}"
#     )

#     documents = loader.load()

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=2000,
#         chunk_overlap=200
#     )

#     texts = text_splitter.split_documents(documents)

#     # Store document chunks in Chroma database
#     vectordb = Chroma.from_documents(
#         documents=texts,
#         embedding=embedding,
#         persist_directory=f"{working_dir}/doc_vectorstore"
#     )

#     return vectordb

def document_process_to_Chroma_db(file_name):

    # Load PDF document
    loader = PyPDFLoader(
        f"{working_dir}/{file_name}"
    )

    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    # Store document chunks in Chroma database
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )

    return vectordb


# def question_answer(user_question):

#     # Load the persisted Chroma vector database
#     vector_db = Chroma(
#         persist_directory=f"{working_dir}/doc_vectorstore",
#         embedding_function=embedding
#     )

#     # Create a retriever for document search
#     retriever = vector_db.as_retriever()

#     # Create RetrievalQA chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#     )

#     # Ask question
#     response = qa_chain.invoke({
#         "query": user_question
#     })

#     answer = response["result"]

#     return answer

def question_answer(user_question):

    # Load the persisted Chroma vector database
    vector_db = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )

    # Create a retriever for document search
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    # Retrieve relevant documents
    documents = retriever.invoke(user_question)

    # Combine retrieved document content
    context = "\n\n".join(
        document.page_content for document in documents
    )

    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context.

    Context:
    {context}

    Question:
    {question}

    If the answer is not available in the context,
    say "I don't know based on the provided document."
    """)

    # Create messages
    messages = prompt.invoke({
        "context": context,
        "question": user_question
    })

    # Get answer from LLM
    response = llm.invoke(messages)

    return response.content