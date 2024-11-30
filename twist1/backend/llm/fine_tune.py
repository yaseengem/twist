import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import PyPDF2

# Function to extract text from PDF files
def extract_text_from_pdfs(pdf_dir):
    texts = []
    for pdf_file in os.listdir(pdf_dir):
        with open(os.path.join(pdf_dir, pdf_file), 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                texts.append(page.extract_text())
    return texts

# Preprocess PDF files
pdf_dir = "./training_data"
texts = extract_text_from_pdfs(pdf_dir)

# Create a dataset from the extracted texts
dataset = Dataset.from_dict({"text": texts})

# Split the dataset into train and test sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download the LLaMA 3.2 1B model and tokenizer
model_name = "meta-llama/LLaMA-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

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
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")