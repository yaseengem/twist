import os
import chromadb
from chromadb.config import Settings
from langchain.chains import SimpleChain
from langchain.prompts import PromptTemplate
import ollama
import fitz  # PyMuPDF

# Initialize Chroma
chroma_client = chromadb.Client(Settings())
collection = chroma_client.create_collection(name="my_collection")

# Load data from the directory
data_directory = "llm/data"
documents = []

for root, _, files in os.walk(data_directory):
    for filename in files:
        if filename.endswith(".txt"):
            with open(os.path.join(root, filename), 'r') as file:
                documents.append(file.read())
        elif filename.endswith(".pdf"):
            pdf_path = os.path.join(root, filename)
            pdf_document = fitz.open(pdf_path)
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            documents.append(text)

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