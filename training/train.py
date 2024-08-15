import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from scrape.DBManager import DBManager
from datasets import Dataset

# Configuration
max_seq_length = 32000
model_name = "unsloth/Meta-Llama-3.1-8B"

# Clear CUDA cache and collect garbage
torch.cuda.empty_cache()
gc.collect()

# Print initial GPU memory usage
print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Define prompt template
comedian_prompt = """You are a comedian and an assistant. Be as funny as possible and help write jokes. Below is a transcript from a standup comedy routine by {comedian}. Study it and learn from its style and content.
### Transcript:
{transcript}
### Instruction:
Based on the style and content of the above transcript, generate a new joke or funny story in the style of {comedian}.
### Response:
"""
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for name, transcript in zip(examples['name'], examples['text']):
        text = comedian_prompt.format(comedian=name, transcript=transcript) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Load transcripts from the database
db_manager = DBManager("comedians.db")
comedian_data = db_manager.get_all_transcripts(tokenizer, max_seq_length - 1000)  # Leave some room for prompt
dataset = Dataset.from_dict({
    "name": [item["name"] for item in comedian_data],
    "text": [item["text"] for item in comedian_data]
})
dataset = dataset.map(formatting_prompts_func, batched=True)

# Use a smaller subset of data if necessary
dataset = dataset.select(range(min(len(dataset), 1000)))  # Adjust the number as needed

# Set up the trainer
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    max_steps=50,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=1,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="constant",
    save_steps=10,
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=max_seq_length,
    dataset_text_field="text",
)

# Disable model caching
model.config.use_cache = False

# Print GPU memory usage before training
print(f"GPU memory allocated before training: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"GPU memory reserved before training: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")

# Train the model
trainer_stats = trainer.train()

# Print training stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")

# Save the model
model.save_pretrained("comedian_lora_model_32k")
tokenizer.save_pretrained("comedian_lora_model_32k")
