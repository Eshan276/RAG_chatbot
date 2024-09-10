from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import openai  # Or another LLM library

app = FastAPI()
# add cors

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))
collection = chroma_client.create_collection(name="documents")

# Initialize OpenAI (replace with your preferred LLM)
openai.api_key = "your-api-key"


class Message(BaseModel):
    content: str


@app.post("/chat")
async def chat(message: Message):
    # Search ChromaDB for relevant documents
    results = collection.query(
        query_texts=[message.content],
        n_results=2
    )

    context = " ".join(results['documents'][0])

    # Generate response using LLM
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. Use this context to inform your responses: {context}"},
            {"role": "user", "content": message.content}
        ]
    )

    return {"response": response.choices[0].message['content']}


@app.post("/add_document")
async def add_document(document: Message):
    # Add document to ChromaDB
    collection.add(
        documents=[document.content],
        ids=[f"doc_{collection.count() + 1}"]
    )
    return {"status": "Document added successfully"}
