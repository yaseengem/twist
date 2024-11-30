import os
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer, TFTrainer, TFTrainingArguments
from datasets import Dataset
import PyPDF2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
logger.info(f"Extracting text from PDFs in directory: {pdf_dir}")
texts = extract_text_from_pdfs(pdf_dir)
logger.info(f"Extracted {len(texts)} texts from PDFs")

# Create a dataset from the extracted texts
dataset = Dataset.from_dict({"text": texts})
logger.info("Created dataset from extracted texts")

# Split the dataset into train and test sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]
logger.info("Split dataset into train and test sets")

# Check if GPU is available
device = tf.config.list_physical_devices('GPU')
if device:
    tf.config.experimental.set_memory_growth(device[0], True)
    logger.info("GPU is available and memory growth is set")
else:
    logger.info("GPU is not available, using CPU")

# Download the LLaMA 3.2 1B model and tokenizer
model_name = "meta-llama/LLaMA-3.2-1B"
logger.info(f"Loading model and tokenizer: {model_name}")
model = TFAutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Set padding token to EOS token")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

logger.info("Tokenizing train and test datasets")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TFTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduce batch size
    per_device_eval_batch_size=1,   # Reduce batch size
    gradient_accumulation_steps=8,  # Accumulate gradients
    fp16=True,                      # Use mixed precision training
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)
logger.info("Defined training arguments")

# Tokenize the dataset with a reduced max length and include labels
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

logger.info("Tokenizing train and test datasets with labels")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define a compute_metrics function (optional)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = tf.argmax(logits, axis=-1)
    return {"accuracy": tf.reduce_mean(tf.cast(predictions == labels, tf.float32)).numpy()}

# Initialize the Trainer
logger.info("Initializing the Trainer")
trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,  # Optional
)

# Train the model
logger.info("Starting model training")
trainer.train()
logger.info("Model training completed")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
logger.info("Saved the fine-tuned model and tokenizer")