import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not detected; training on CPU will be very slow.")

# Configuration
model_name = "distilgpt2"  # A small model for demonstration
processed_file = "processed_data_large.csv"

try:
    df = pd.read_csv(processed_file)
    print(f"Loaded {processed_file} successfully.")
    df["full_text"] = "Review: " + df["reviewText"].astype(str) + " Sentiment: " + df["sentiment"].astype(str)
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
except Exception as e:
    print(f"Could not load file '{processed_file}'. Creating a dummy dataset instead.")
    dummy_data = {"full_text": [
        "Review: amazing product. Sentiment: positive", 
        "Review: terrible service. Sentiment: negative"
    ]}
    df = pd.DataFrame(dummy_data)
    train_df, eval_df = train_test_split(df, test_size=0.5, random_state=42)

# Initialize tokenizer and set pad token if missing
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.to(device)

# Setup PEFT (LoRA) configuration.
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # For GPT-2 style models, commonly the attention projection
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print("Trainable parameters:")
print(model.print_trainable_parameters())

# Disable caching for training.
model.config.use_cache = False

# Define a custom Trainer that accepts extra kwargs in compute_loss.
class ColabTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Tokenization function: tokenize the texts and add labels.
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["full_text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    # Copy input_ids to labels for causal LM objective.
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# Create Hugging Face datasets from pandas DataFrames.
train_dataset = Dataset.from_pandas(train_df[["full_text"]])
eval_dataset = Dataset.from_pandas(eval_df[["full_text"]])

# Tokenize datasets.
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["full_text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["full_text"])

# Training arguments: disable gradient_checkpointing.
training_args = TrainingArguments(
    output_dir="./distilgpt2_sentiment",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    gradient_checkpointing=False,  
)

# Initialize the trainer.
trainer = ColabTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Begin training.
print("Starting training...")
trainer.train()
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
print("Training successful!")