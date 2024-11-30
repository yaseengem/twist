from fastapi import FastAPI
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Create FastAPI app
app = FastAPI()

@app.post("/generate")
async def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8000)