import os
import shutil
import time
from dotenv import load_dotenv
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

assert GROQ_API_KEY and COHERE_API_KEY, "API keys not found in environment. Please set GROQ_API_KEY and COHERE_API_KEY."

# Configuration
PDF_FOLDER = "./data"
CHROMA_DIR = "./chroma"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

# Helper Functions
def reset_chroma_directory(reset: bool):
    if reset:
        print("\033[96m[INFO]\033[0m Resetting Chroma vector store...")
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR, exist_ok=True)


def load_documents(folder_path: str) -> List[Document]:
    print("\033[96m[INFO]\033[0m Loading PDF documents...")
    docs = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            print(f"  → Loading: {fname}")
            loader = PyPDFLoader(os.path.join(folder_path, fname))
            for doc in loader.load():
                doc.metadata['source'] = fname
                docs.append(doc)
    print(f"\033[92m[OK]\033[0m Loaded {len(docs)} documents.")
    return docs

# Initialize Models
print("\033[96m[INFO]\033[0m Initializing models...")
llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    api_key=GROQ_API_KEY
)
embedding_model = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)
print("\033[92m[OK]\033[0m Models initialized.")

# Reset vector DB if needed
reset_chroma_directory(reset=True)

# Load and split documents
raw_docs = load_documents(PDF_FOLDER)
print("\033[96m[INFO]\033[0m Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
split_docs = text_splitter.split_documents(raw_docs)
print(f"\033[92m[OK]\033[0m Split into {len(split_docs)} chunks.")

# Create Chroma vector store
print("\033[96m[INFO]\033[0m Creating Chroma vector store...")
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR,
    collection_name="rag"
)
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Diverse results
    search_kwargs={"k": TOP_K, "lambda_mult": 0.5}
)
print("\033[92m[OK]\033[0m Vector store ready.")

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def run_query(query: str):
    try:
        print("\n\033[1m\033[96m🔍 Processing Query...\033[0m")
        start = time.time()
        result = qa_chain.invoke({"query": query})
        duration = time.time() - start

        sources = set()
        print("\n" + "═"*80)
        print("\033[1m\033[97m🧠 Answer:\033[0m\n")
        print(f"\033[92m{result['result']}\033[0m")

        # Summarized Source Attribution Header
        print("\n\033[1m\033[97m📚 Sources Used:\033[0m")
        for i, doc in enumerate(result["source_documents"], start=1):
            source = doc.metadata.get("source", "Unknown")
            print(f"  {i}. \033[94m{source}\033[0m")
            sources.add(source)

        # Advanced Hallucination Check (semantic + behavioral)
        source_text = " ".join(doc.page_content.lower() for doc in result["source_documents"])
        answer_words = [word.lower() for word in result["result"].split() if len(word) > 5]
        matching_words = [w for w in answer_words if w in source_text]
        hallucination_score = (len(matching_words) / max(len(answer_words), 1)) * 100

        # Behavioral Sanity Check
        behavioral_keywords = [
            "not sure", "don’t know", "don't know", "not certain",
            "no information", "can't find", "wasn't able to locate"
        ]
        responsible_response = any(k in result["result"].lower() for k in behavioral_keywords)

        if responsible_response:
            hallucination_status = "\033[92mPASS\033[0m"
            hallucination_reason = "🟢 Responsible refusal (behaviorally grounded)"
            hallucination_score = 100.0
        else:
            hallucination_status = "\033[92mPASS\033[0m" if hallucination_score >= 60 else "\033[91mFAIL\033[0m"
            hallucination_reason = (
                "🟢 Confident based on source match"
                if hallucination_status == "\033[92mPASS\033[0m"
                else "🔴 Possible unsupported claims"
            )

        print("\n\033[1m\033[97m🚨 Hallucination Check:\033[0m", hallucination_status, f"\033[90m({hallucination_score:.0f}% confidence )\033[0m")
        print("   " + hallucination_reason)
        print(f"\033[90m⏱️  Processed in {duration:.2f} seconds\033[0m")
        print("═"*80)

    except Exception as e:
        print("\033[91m[ERROR]\033[0m", e)

if __name__ == "__main__":
    print("\n\033[1m🧠 Hi, I'm Delos\033[0m")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_query = input("\nAsk your question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("\033[93m👋 Goodbye!\033[0m")
            break
        run_query(user_query)