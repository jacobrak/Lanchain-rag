from dotenv import load_dotenv
import os
import faiss
import numpy as np
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global enviorment
# Ladda PDF
loader = DirectoryLoader("./data", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
assert len(docs) > 0, "failed to load documents"

# Openai embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
embedded_docs = embeddings.embed_documents([doc.page_content for doc in docs])
assert len(embedded_docs) > 0, "failed to embedded_docs documents"


# Faiss
dimension = len(embedded_docs[0])  # Embedding dimension
faiss_index = faiss.IndexFlatL2(dimension)  #

# Convert to numpy arrays
np_embeddings = np.array(embedded_docs).astype('float16') # float16 for faster computation

# Add the embeddings to FAISS index
faiss_index.add(np_embeddings)
text_embedding_pairs = [(doc.page_content, embedding) for doc, embedding in zip(docs, embedded_docs)]

#  Set Up Retriever
vector_store = FAISS.from_embeddings(text_embedding_pairs, embeddings)  # Store embeddings
retriever = vector_store.as_retriever()  # Convert FAISS embedding to retriever


# Run LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)



def hallucination_node(answer: str, retrieved_docs):
    """
    Checks if the generated answer is consistent
    Flags the answer if no supporting evidence is found
    """
    print(retrieved_docs)
    print(type(retrieved_docs))
    evidence_found = False
    for doc in retrieved_docs:
        if answer.lower() in doc.page_content.lower():  # Check if answer is in retrieved docs
            evidence_found = True
            break
    
    if evidence_found:
        return "Answer Verified"
    else:
        return "Hallucination Detected"


def ask_question(query):
    """Function to query the RAG system and get an answer."""
    retrieved_docs = retriever.get_relevant_documents(query)  # Retrieve docs
    response = qa_chain.run(query)  # Get answer 
    
    # Hallucination_node
    hallucination_result = hallucination_node(response, retrieved_docs)

    print("")
    print(f"Question: {query}")
    print(f"Hallucination check: {hallucination_result}")
    print("")

    return response
def ask_question_running():
    """Function to query the RAG system and get an answer."""
    while True:
        query = input("")
        if query.lower() == "exit":
            return False
        retrieved_docs = retriever.get_relevant_documents(query)  # Retrieve docs
        response = qa_chain.run(query)  # Get answer 
        
        # Hallucination_node
        hallucination_result = hallucination_node(response, retrieved_docs)

        print("")
        print(f"Question: {query}")
        print(f"Hallucination check: {hallucination_result}")
        print("")

        return response 
    
def ask_question_test(query):
    """Function to query the RAG system and get an answer."""
    retrieved_docs = retriever.get_relevant_documents(query)  # Retrieve docs
    response = qa_chain.run(query)  # Get answer 
    
    # Hallucination_node
    hallucination_result = hallucination_node(response, retrieved_docs)

    return hallucination_result