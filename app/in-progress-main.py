import getpass
import os
import uuid
import sys
import json
from langserve import add_routes
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain.chains import LLMChain

import weaviate
from weaviate import auth, connect, WeaviateClient

from minio import Minio

os.environ["OPENAI_API_KEY"] = ""

# Configuration
class AppConfig:
    pass

class LangchainConfig:
    pass

class LLMConfig:
    API_KEY = ""

class ToolConfig:
    WEAVIATE_ENDPOINT = "http://192.168.0.21:8080"
    WEAVIATE_API_KEY = ""
    MINIO_ENDPOINT = "play.min.io:443"
    MINIO_ACCESS_KEY = "minioadmin"
    MINIO_SECRET_KEY = "minioadmin"

app_config = AppConfig()
langchain_config = LangchainConfig()
llm_config = LLMConfig()
tool_config = ToolConfig()

class MinioOperations:
    name = "MinIO"
    description = "A tool for interacting with MinIO object storage."

    def __init__(self, minio_url, access_key, secret_key, secure=False):
        self.config = {
            "endpoint": minio_url,
            "access_key": access_key,
            "secret_key": secret_key,
            "secure": secure
        }

    def load_documents_from_minio(self, bucket_name):
        documents = []
        objects = self.client.list_objects(bucket_name)

        for obj in objects:
            document = self.client.get_object(bucket_name, obj.object_name)
            content = document.read().decode("utf-8")
            documents.append(content)
            document.close()

        return documents

    def upload_document(self, bucket_name, document_name, document):
        self.client.put_object(
            bucket_name,
            document_name,
            document,
            length=len(document),
            content_type="application/text"
        )

    def delete_document(self, bucket_name, document_name):
        self.client.remove_object(bucket_name, document_name)

    def list_documents(self, bucket_name):
        documents = []
        objects = self.client.list_objects(bucket_name)

        for obj in objects:
            documents.append(obj.object_name)

        return documents

# WeaviateTool
class WeaviateTool(BaseTool):
    name = "Weaviate"
    description = "A tool for interacting with Weaviate vector database."

    def __init__(self, weaviate_url, api_key):
        self.config = {
            "url": weaviate_url,
            "api_key": api_key
        }

    def run(self, query: str) -> str:
        # Implement Weaviate operations here
        pass

# WeaviateOperations
class WeaviateOperations:
    def __init__(self, weaviate_url, api_key=None):
        # Updated to use WeaviateClient and optional API key for authentication
        connection_params = connect.ConnectionParams(
            http_host=weaviate_url,
            http_secure=True if "https://" in weaviate_url else False
        )
        if api_key:
            auth_client_secret = auth.AuthApiKey(api_key)
        else:
            auth_client_secret = None
        self.client = WeaviateClient(connection_params, auth_client_secret)
        self.client.connect()  # Explicitly connect

    def index_document(self, collection_name, document_uuid, document):
        # Adapted to use collections and data objects
        collection = self.client.collections.get(collection_name)
        if not collection:
            # Create a collection if it doesn't exist (simplified version)
            self.client.collections.create_from_dict({
                "name": collection_name,
                "description": f"Documents from {collection_name}",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content"
                    }
                ]
            })
        # Index the document
        collection.data.insert(properties=document, uuid=document_uuid)

    def query_data(self, query):
        # Implement data querying logic using the updated query methods
        result = self.client.query.get(collection="Document", properties=["content"]) \
            .with_near_text({"concepts": [query]}).do()
        return [obj["content"] for obj in result["data"]["Get"]["Document"]]

    def update_document(self, collection_name, document_uuid, update_properties):
        # Updated document updating logic
        self.client.collections.get(collection_name).data.update(
            properties=update_properties,
            uuid=document_uuid
        )

    def delete_document(self, collection_name, document_uuid):
        # Updated document deletion logic
        self.client.collections.get(collection_name).data.delete(uuid=document_uuid)

    def close(self):
        # Close the client connection explicitly when done
        self.client.close()

# MinioOperations
def load_documents_from_minio(bucket_name, minio_endpoint, minio_access_key, minio_secret_key):
    # Implement logic to load documents from MinIO bucket here
    pass

# Document Processing and Indexing
document_processing_prompt = PromptTemplate.from_template("Analyze and summarize the main points of the following document for efficient indexing and retrieval: {document_content}")
document_processing_model = ChatOpenAI()
document_processing_chain = document_processing_prompt | document_processing_model

# Query Handling
query_handling_prompt = PromptTemplate.from_template("Convert this user query into an actionable search command for database retrieval: {user_query}")
query_handling_chain = query_handling_prompt | document_processing_model

# Document Updating
document_updating_prompt = PromptTemplate.from_template("Given the document ID {uuid} and update details {update_information}, generate a summary of changes to be applied.")
document_updating_chain = document_updating_prompt | document_processing_model

# Document Deletion
document_deletion_prompt = PromptTemplate.from_template("Provide a rationale for deleting the document with ID {uuid}, considering its content and relevance.")
document_deletion_chain = document_deletion_prompt | document_processing_model

app = FastAPI(
    title="Cdaprod AI API Gateway",
    version="1.0",
    description="An api server using Langchain's Runnable interfaces",
)

# Weaviate and MinIO connection details, ensure these are correctly configured
WEAVIATE_ENDPOINT = "http://192.168.0.21:8080"
MINIO_ENDPOINT = "play.min.io:443"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "cda-dataset"

# Locally configured LLM via Ollama
llama = Ollama(model="llama2")

# Pydantic models
class MinioEvent(BaseModel):
    eventName: str
    bucket: dict
    object: dict

class MinioBucket(BaseModel):
    name: str
    owner: Optional[str] = None
    creationDate: Optional[str] = None

class MinioObject(BaseModel):
    key: str
    size: int
    contentType: Optional[str] = None
    metadata: Optional[dict] = Field(default_factory=dict)
    bucket: Optional[MinioBucket] = None

class MinioPath(BaseModel):
    path: str
    parentPath: Optional[str] = None
    objects: List[MinioObject] = Field(default_factory=list)

class MinioFile(BaseModel):
    key: str
    size: int
    contentType: Optional[str] = "application/octet-stream"
    metadata: Optional[dict] = Field(default_factory=dict)
    content: Optional[str] = None
    bucket: Optional[MinioBucket] = None

# Langchain runnables 
class DocumentProcessingRunnable(Runnable):
    def __init__(self, minio_tool, weaviate_tool):
        self.llm = ChatOpenAI(api_key=llm_config.API_KEY)
        self.weaviate_ops = WeaviateOperations(weaviate_tool.config['url'])
        self.bucket_name = MINIO_BUCKET
        self.minio_ops = lambda: load_documents_from_minio(self.bucket_name, minio_tool.config['endpoint'], minio_tool.config['access_key'], minio_tool.config['secret_key'])

    def run(self, _):
        documents = self.minio_ops()

        for doc in documents:
            processed_doc = self.process_document(doc)
            doc_name = self.extract_document_name(processed_doc)
            self.weaviate_ops.index_document(self.bucket_name, doc_name, processed_doc)

    def process_document(self, document):
        prompt = f"Process this document: {document}"
        response = self.llm.complete(prompt=prompt)
        return response

    def extract_document_name(self, document):
        return "ExtractedDocumentName"

def initialize_tools():
    minio_tool = MinioTool(minio_url=tool_config.MINIO_ENDPOINT,
                           access_key=tool_config.MINIO_ACCESS_KEY,
                           secret_key=tool_config.MINIO_SECRET_KEY,
                           secure=False)
    weaviate_tool = WeaviateTool(tool_config.WEAVIATE_ENDPOINT, tool_config.WEAVIATE_API_KEY)
    return minio_tool, weaviate_tool

# Wrapping our LLM and FastAPI app with Langserve routes 
add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

add_routes(
    app,
    llama(),
    path="/llama2",
)

# Additional Anthropic LLM + Prompting(designed around the joke chat template) + Langserve Route 
# model = ChatAnthropic()
prompt = ChatPromptTemplate.from_template("Fetch")

# Expose chains as endpoints with LangServe
add_routes(app, document_processing_chain, path="/process-document")
add_routes(app, query_handling_chain, path="/handle-query")
add_routes(app, document_updating_chain, path="/update-document")
add_routes(app, document_deletion_chain, path="/delete-document")

# Our FASTAPI Routes that return our runnables and other logic
@app.get("/")
async def root():
    return {"message": "LangChain-Weaviate-MinIO Integration Service"}

@app.post("/index_from_minio")
async def index_from_minio():
    minio_tool, weaviate_tool = initialize_tools()
    runnable = DocumentProcessingRunnable(minio_tool, weaviate_tool)
    runnable.run(None)
    return {"status": "Indexing complete"}

@app.post("/query")
async def query_weaviate(query: str):
    _, weaviate_tool = initialize_tools()
    weaviate_ops = WeaviateOperations(weaviate_tool.config['url'])
    return weaviate_ops.query_data(query)

@app.post("/update/{uuid}")
async def update_document(uuid: str, update_properties: dict):
    _, weaviate_tool = initialize_tools()
    weaviate_ops = WeaviateOperations(weaviate_tool.config['url'])
    weaviate_ops.update_document(uuid, update_properties)
    return {"status": "Document updated"}

@app.delete("/delete/{uuid}")
async def delete_document(uuid: str):
    _, weaviate_tool = initialize_tools()
    weaviate_ops = WeaviateOperations(weaviate_tool.config['url'])
    weaviate_ops.delete_document(uuid)
    return {"status": "Document deleted"}

@app.post("/minio-event")
async def handle_minio_event(event: MinioEvent):
    bucket_name = event.bucket["name"]
    object_key = event.object["key"]
    object_size = event.object["size"]
    content_type = event.object.get("contentType", "application/octet-stream")

    minio_tool, weaviate_tool = initialize_tools()
    runnable = DocumentProcessingRunnable(minio_tool, weaviate_tool)
    runnable.run(None)
    
    return {"message": "Event processed successfully"}

# The main executor for the application that keeps it running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)