# llm/start_ollama_server.sh

#!/bin/bash

# Ensure Ollama CLI is installed and configured
if ! command -v ollama &> /dev/null
then
    echo "Ollama CLI could not be found. Please install it from https://ollama.com"
    exit
fi

# Start the Ollama server
ollama serve --model ./fine_tuned_model --port 8000