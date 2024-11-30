# llm/fine_tune_and_host.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from fastapi import FastAPI
import uvicorn

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download the LLaMA 3.2 1B model and tokenizer
model_name = "meta-llama/LLaMA-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load custom dataset
dataset = load_dataset("path/to/your/custom/dataset")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

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