import torch
from datasets import Dataset
from transformers import AutoTokenizer, TextStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from DBManager import DBManager  # Import your DBManager

# Configuration
max_seq_length = 128000
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA (same as before)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

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

# Set up the trainer (same as before)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=SFTTrainer.SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=100,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Train the model
trainer_stats = trainer.train()

# Print training stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")

# Save the model
model.save_pretrained("comedian_lora_model_128k")
tokenizer.save_pretrained("comedian_lora_model_128k")

# Test the model
FastLanguageModel.for_inference(model)
test_prompt = "Generate a comedy routine in the style of the transcripts you were trained on."
inputs = tokenizer([comedian_prompt.format(test_prompt)], return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1000)  # Increased max_new_tokens for longer generation