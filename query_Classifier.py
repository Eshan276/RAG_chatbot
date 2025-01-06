from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
import datetime
from together import Together
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
load_dotenv()
# Set the NLTK data directory
# nltk_data_dir = os.path.join(os.getenv("APPDATA"), "nltk_data")
# if not os.path.exists(nltk_data_dir):
#     os.makedirs(nltk_data_dir)
# nltk.data.path.append(nltk_data_dir)

# # Download necessary NLTK data
# nltk.download('punkt_tab', download_dir=nltk_data_dir)
# nltk.download('stopwords', download_dir=nltk_data_dir)

app = FastAPI()

# Initialize ChromaDB client
client = chromadb.Client()

# Create or get the collection
collection = client.get_or_create_collection("document_collection")

# Initialize text splitters
character_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
token_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)

# Initialize embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")

# Initialize Together AI client
# Replace with your actual API key
together_client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))


# Initialize summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize TF-IDF vectorizer for keyword indexing
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Initialize query classifier
query_classifier_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")
query_classifier_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")

# Initialize keyword index
keyword_index = {}


class Message(BaseModel):
    content: str


class Query(BaseModel):
    query: str
    n_results: int = 5


def preprocess_text(text):
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])


def update_keyword_index(doc_id, text):
    preprocessed_text = preprocess_text(text)
    for word in preprocessed_text.split():
        if word not in keyword_index:
            keyword_index[word] = set()
        keyword_index[word].add(doc_id)


def keyword_search(query, n_results=5):
    preprocessed_query = preprocess_text(query)
    query_words = set(preprocessed_query.split())
    results = {}
    # for a in keyword_index:
    #     print(a)
    for word in query_words:

        if word in keyword_index:
            for doc_id in keyword_index[word]:
                results[doc_id] = results.get(doc_id, 0) + 1
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_results[:n_results]]


def classify_query(query):
    inputs = query_classifier_tokenizer(query, return_tensors="pt")
    outputs = query_classifier_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()
    return "command" if predicted_class == 1 else "natural_language"


@app.post("/add_document")
async def add_document(document: Message):
    try:
        # Split the document text
        character_split_texts = character_splitter.split_text(document.content)
        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)

        token_split_texts = [
            text for text in token_split_texts if isinstance(text, str)]
        if not token_split_texts:
            raise HTTPException(
                status_code=400, detail="No valid tokenized text found.")

        # Generate embeddings
        embeddings = embedding_function(token_split_texts)

        # Create metadata
        metadatas = [{
            "source": "document_upload",
            "date_added": datetime.datetime.now().isoformat(),
            "chunk_index": i
        } for i in range(len(token_split_texts))]

        # Generate IDs
        ids = [
            f"doc_{collection.count() + i + 1}" for i in range(len(token_split_texts))]

        # Add to ChromaDB
        collection.add(
            documents=token_split_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # Update keyword index
        for i, text in enumerate(token_split_texts):
            update_keyword_index(ids[i], text)

        return {
            "status": "Document added successfully",
            "total_chunks": len(token_split_texts)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/query_documents")
async def query_documents(query: Query):
    try:
        query_type = classify_query(query.query)
        print(query_type)
        # result = collection.get(ids=["doc_7"], include=[
        #     "documents"])
        # print(result)
        if query_type == "command":
            # Perform keyword search
            relevant_ids = keyword_search(query.query, query.n_results)
            print(relevant_ids)
            # print all the ids present in chroma dbb

            # qresults = collection.get(
            #     ids=["doc_1"],
            #     include=["documents", "metadatas"]
            # )
            # document_ids = ["doc_1", "doc_2", "doc_3"]
            # presult = collection.get(ids=["doc_7", "doc_1"], include=[
            #     "documents"])
            # print("p", presult)
            # print(type(relevant_ids[0]))
            results = collection.get(ids=relevant_ids, include=[
                                     "documents", "metadatas"])
            print("in", results["documents"][0])
        else:
            # Perform vector search
            query_embedding = embedding_function([query.query])[0]
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=query.n_results,
                include=["documents", "metadatas", "distances"]
            )
            print("t", results)

        # Format the results
        formatted_results = []
        print(len(results['ids']))
        for i in range(len(results['ids'])):
            print("t", results['documents'][i])
            formatted_results.append({
                "document": results['documents'][i],
                "metadata": results['metadatas'][i],
                "distance": results['distances'][i] if 'distances' in results else None
            })

        return {
            "query": query.query,
            "query_type": query_type,
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/chat")
async def chat(message: Message):
    try:
        query_type = classify_query(message.content)

        if query_type == "command":
            # Perform keyword search
            relevant_ids = keyword_search(message.content, 5)
            results = collection.get(
                ids=relevant_ids,
                include=["documents"]
            )
        else:
            # Perform vector search
            results = collection.query(
                query_texts=[message.content],
                n_results=5,
                include=["documents", "distances"]
            )
        print(query_type)
        print(results)
        if not results['documents']:
            raise HTTPException(
                status_code=404, detail="No relevant documents found.")
        print(type(results['documents'][0]))
        if query_type == "command":
            documents = results['documents'][0]
        else:
            documents = [doc for doc in results['documents'][0]
                         if isinstance(doc, str)]
        # print("resut", results['documents'][0])
        # documents = [doc for doc in results['documents'][0]
        #              if isinstance(doc, str)]
        if not documents:
            raise HTTPException(
                status_code=404, detail="No valid documents found in the results.")
        context = ""
        if query_type != "command":
            context = " ".join(documents[:2])
        else:
            context = documents
        print("doc", documents)
        print(context)
        summarized_context = summarizer(
            context, max_length=100, min_length=25, do_sample=False
        )

        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Use this context to inform your responses: "
                f"{{'summarized context': '{summarized_context[0]['summary_text']}', 'context': '{context}'}}"
            },
            {
                "role": "user",
                "content": message.content
            }
        ]
        print(messages)

        response = together_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"]
        )

        return {
            "response": response.choices[0].message.content,
            "query_type": query_type
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
