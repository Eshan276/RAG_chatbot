from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from dotenv import load_dotenv
from together import Together
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
import datetime

# Load environment variables
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection")

# Initialize SentenceTransformer embedding function
embedding_function = SentenceTransformerEmbeddingFunction()
# Initialize Together AI client
client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

# Optional: Initialize a summarization model
summarizer = pipeline("summarization")
# Initialize text splitters
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)


class Message(BaseModel):
    content: str
    type: str = "chat"


# app.mount("/static", StaticFiles(directory="."), name="static")


# @app.get("/chatbot.html", response_class=HTMLResponse)
# async def get_chatbot_html():
#     with open("chatbot.html", "r") as file:
#         return HTMLResponse(content=file.read(), status_code=200)


# @app.post("/chat")
# async def chat(message: Message):
#     try:
#         # Retrieve relevant documents
#         results = collection.query(query_texts=[message.content], n_results=5)

#         if not results['documents']:
#             return {"error": "No relevant documents found."}

#         # Ensure results['documents'] are strings
#         documents = [doc for doc in results['documents']
#                      if isinstance(doc, str)]
#         if not documents:
#             return {"error": "No valid documents found in the results."}

#         # Preprocess the retrieved documents
#         context = " ".join(documents[:2])
#         summarized_context = summarizer(
#             context, max_length=100, min_length=25, do_sample=False
#         )

#         # Construct messages for Together AI
#         emessages = [
#             {"role": "system",
#                 "content": f"You are a helpful assistant. Use this context to inform your responses: {summarized_context[0]['summary_text']}"},
#             {"role": "user", "content": message.content}
#         ]

#         # Generate response using Together AI
#         response = client.chat.completions.create(
#             model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
#             messages=emessages,
#             max_tokens=512,
#             temperature=0.7,
#             top_p=0.7,
#             top_k=50,
#             repetition_penalty=1,
#             stop=["<|eot_id|>"]
#         )

#         return {"response": response.choices[0].message.content}

#     except Exception as e:
#         return {"error": str(e)}
@app.post("/chat")
async def chat(message: Message):
    try:
        # Retrieve relevant documents
        results = collection.query(
            query_texts=[message.content],
            n_results=5,
            include=["documents"]
        )

        if not results['documents']:
            raise HTTPException(
                status_code=404, detail="No relevant documents found.")

        # Ensure results['documents'] are strings
        documents = [doc for doc in results['documents']
                     [0] if isinstance(doc, str)]
        if not documents:
            raise HTTPException(
                status_code=404, detail="No valid documents found in the results.")
        print(documents)
        # Preprocess the retrieved documents
        context = " ".join(documents[:2])
        if (message.type == "summarize"):
            summarized_context = summarizer(
                context, max_length=100, min_length=25, do_sample=False
            )
        else:
            summarized_context = [{"summary_text": context}]

        # Construct messages for Together AI
        messages = [
            {"role": "system",
             "content": f"You are a helpful assistant. Use this context to inform your responses: {summarized_context[0]['summary_text']}"},
            {"role": "user", "content": message.content}
        ]
        print(messages)
        # Generate response using Together AI
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"]
        )

        return {"response": response.choices[0].message.content}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")

character_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
token_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)

# Initialize embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")


@app.post("/add_document")
async def add_document(document: Message):
    try:
        # Step 1: Split the document text
        character_split_texts = character_splitter.split_text(document.content)

        # Step 2: Further split text into tokens
        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)

        # Ensure token_split_texts are strings
        token_split_texts = [
            text for text in token_split_texts if isinstance(text, str)]
        if not token_split_texts:
            raise HTTPException(
                status_code=400, detail="No valid tokenized text found.")

        # Step 3: Generate embeddings for each tokenized chunk
        embeddings = embedding_function(token_split_texts)

        # Step 4: Create metadata for each chunk
        metadatas = [{
            "source": "document_upload",
            "date_added": datetime.datetime.now().isoformat(),
            "chunk_index": i
        } for i in range(len(token_split_texts))]

        # Step 5: Generate IDs for each chunk
        ids = [
            f"doc_{collection.count() + i + 1}" for i in range(len(token_split_texts))]

        # Step 6: Add chunks, embeddings, and metadata to ChromaDB
        collection.add(
            documents=token_split_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # Step 7: Verify the addition by retrieving a sample
        sample = collection.get(
            ids=[ids[0]],
            include=["documents", "embeddings", "metadatas"]
        )

        return {
            "status": "Document added successfully",
            "sample": sample,
            "total_chunks": len(token_split_texts)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/get_all_documents")
async def get_all_documents():
    coll = chroma_client.get_collection("my_collection")
    res = coll.get()
    print(res)
    return coll.documents


@app.get("/get_documents")
async def get_documents():
    try:
        # Retrieve all documents with their embeddings and metadata
        results = collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


class Query(BaseModel):
    query: str
    n_results: int = 5


@app.post("/query_documents")
async def query_documents(query: Query):
    try:
        # Generate embedding for the query text
        query_embedding = embedding_function([query.query])[0]

        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Format the results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })

        return {
            "query": query.query,
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")
