import os
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from together import Together
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
load_dotenv()
# Initialize ChromaDB
chroma_client = chromadb.Client()
# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")
print(os.environ.get('TOGETHER_API_KEY'))
# Initialize Together AI client
client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))


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
    print(context)
    emessages = [
        {"role": "system", "content": f"You are a helpful assistant. Use this context to inform your responses: {context}"},
        {"role": "user", "content": message.content}
    ]
    print(emessages)
    # Generate response using Together AI
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. Use this context to inform your responses: {context}"},
            {"role": "user", "content": message.content}
        ],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>"]
    )

    return {"response": response.choices[0].message.content}


@app.post("/add_document")
async def add_document(document: Message):
    # Add document to ChromaDB
    collection.add(
        documents=[document.content],
        ids=[f"doc_{collection.count() + 1}"]
    )
    return {"status": "Document added successfully"}


# curl - -location 'http://localhost:8000/add_document' \
#     - -header 'Content-Type: application/json' \
#     - -data '{
#         "content": "my name is eshan"
#     }'
