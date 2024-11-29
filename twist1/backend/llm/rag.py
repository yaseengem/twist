import os
import chromadb
from chromadb.config import Settings
from langchain.chains import SimpleChain
from langchain.prompts import PromptTemplate
import ollama

# Initialize Chroma
chroma_client = chromadb.Client(Settings())
collection = chroma_client.create_collection(name="my_collection")

# Load data from the directory
data_directory = "/path/to/your/data"
documents = []
for filename in os.listdir(data_directory):
    if filename.endswith(".txt"):
        with open(os.path.join(data_directory, filename), 'r') as file:
            documents.append(file.read())

# Example embeddings (replace with actual embeddings)
embeddings = [[0.1, 0.2, 0.3] for _ in documents]

# Add documents and embeddings to the collection
collection.add(documents=documents, embeddings=embeddings)

# Define a simple chain
prompt_template = PromptTemplate(template="Translate the following text to French: {text}")
chain = SimpleChain(prompt_template=prompt_template)

# Initialize Ollama with the Llama model
ollama_client = ollama.Client(model="llama3.2-1B")

# Process each document
for document in documents:
    response = ollama_client.generate(prompt=f"Translate '{document}' to French.")
    print(response)