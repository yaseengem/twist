# # llm/setup_environment.sh

# #!/bin/bash

# # Function to check if a command exists
# command_exists() {
#     command -v "$1" >/dev/null 2>&1
# }

# # Update package list
# sudo apt-get update

# # Install Python3 and pip if not installed
# if ! command_exists python3; then
#     sudo apt-get install -y python3
# fi

# if ! command_exists pip3; then
#     sudo apt-get install -y python3-pip
# fi

# # Install PyTorch
# pip3 install torch torchvision torchaudio

# # Install Hugging Face Transformers and Datasets
# pip3 install transformers datasets

# # Install FastAPI and Uvicorn for serving endpoints
# pip3 install fastapi uvicorn

# # Install Ollama CLI
# if ! command_exists ollama; then
#     # Assuming Ollama CLI can be installed via a script or package manager
#     # Replace the following line with the actual installation command for Ollama CLI
#     echo "Please install Ollama CLI manually from https://ollama.com"
# fi

# echo "All necessary packages are installed."