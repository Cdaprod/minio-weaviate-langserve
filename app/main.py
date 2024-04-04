import os
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langserve import add_routes
from minio import Minio
from weaviate import Client

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

app = FastAPI()

# Initialize Minio client
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# Initialize Weaviate client
weaviate_client = Client("http://weaviate:8080")

# Define prompt template for document processing
document_processing_prompt = PromptTemplate.from_template("Summarize the following document: {document}")
document_processing_chain = document_processing_prompt | ChatOpenAI()

# Define prompt template for query processing
query_processing_prompt = PromptTemplate.from_template("Generate a search query for the following question: {question}")
query_processing_chain = query_processing_prompt | ChatOpenAI()

# Langserve route for document processing
add_routes(app, document_processing_chain, path="/process-document")

# Langserve route for query processing
add_routes(app, query_processing_chain, path="/process-query")

@app.post("/upload-document")
async def upload_document(document: str):
    # Upload document to Minio
    minio_client.put_object("documents", "document.txt", document, len(document))

    # Process document using Langchain
    processed_document = document_processing_chain.run(document)

    # Index processed document in Weaviate
    weaviate_client.data_object.create(
        data_object={"name": "document", "content": processed_document},
        class_name="Document"
    )

    return {"message": "Document uploaded and processed successfully"}

@app.post("/search-documents")
async def search_documents(question: str):
    # Process query using Langchain
    search_query = query_processing_chain.run(question)

    # Search documents in Weaviate
    result = weaviate_client.query.get("Document", ["content"]).with_where({"path": ["content"], "operator": "Contains", "valueText": search_query}).do()

    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)