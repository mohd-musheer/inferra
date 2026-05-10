# Inferra-Qwen-LoRA

Lightweight conversational AI model built using QLoRA fine-tuning on top of Qwen2.5-3B-Instruct using consumer-grade hardware.

---

# Overview

Inferra-Qwen-LoRA is a parameter-efficient fine-tuned conversational AI model designed for:
- Instruction following
- Conversational responses
- Lightweight reasoning
- Fast inference
- Low VRAM deployment

The project demonstrates practical LLM fine-tuning using:
- QLoRA
- LoRA adapters
- 4-bit quantization
- Unsloth acceleration
- Hugging Face ecosystem

---

# Base Model

Base Model:
Qwen/Qwen2.5-3B-Instruct

Frameworks:
- Transformers
- PEFT
- Unsloth
- BitsAndBytes

---

# Features

- QLoRA fine-tuning
- 4-bit quantized inference
- Low VRAM usage
- Fast response generation
- Hugging Face compatible
- PEFT adapter architecture
- Consumer GPU friendly
- Docker deployment support
- FastAPI serving support

---

# Model Architecture

```text
Base Qwen Model
       +
LoRA Adapter
       =
Final Inference Model
```

Only LoRA adapter weights are trained.

This reduces:
- Training cost
- VRAM usage
- Deployment size
- Storage requirements

---

# Training Details

| Property | Value |
|---|---|
| Base Model | Qwen2.5-3B-Instruct |
| Fine-Tuning Method | QLoRA |
| Precision | 4-bit |
| Adapter Type | LoRA |
| Trainable Parameters | ~15M |
| Total Parameters | ~3.1B |
| GPU Used | NVIDIA T4 |
| Platform | Kaggle |
| Training Type | Conversational SFT |

---

# Why QLoRA?

QLoRA allows efficient LLM fine-tuning without training the full model.

Advantages:
- Lower VRAM usage
- Faster experimentation
- Smaller model uploads
- Affordable fine-tuning
- Faster deployment

---

# Hardware Used

| Component | Value |
|---|---|
| GPU | NVIDIA T4 |
| VRAM | 15GB |
| Platform | Kaggle |
| CUDA | Enabled |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/inferra-qwen-lora.git

cd inferra-qwen-lora
```

---

# Install Dependencies

```bash
pip install -U transformers accelerate peft bitsandbytes unsloth
```

---

# Hugging Face Model

```text
mohdmusheer/inferra
```

---

# Load Model

```python
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = PeftModel.from_pretrained(
    model,
    "mohdmusheer/inferra",
)

FastLanguageModel.for_inference(model)

print("Model Loaded Successfully!")
```

---

# Inference Example

```python
messages = [
    {
        "role": "user",
        "content": "Explain machine learning in simple words."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

generated_tokens = outputs[0][inputs.input_ids.shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print(response)
```

---

# Training Configuration

```python
max_seq_length = 1024

per_device_train_batch_size = 1

gradient_accumulation_steps = 8

max_steps = 500

learning_rate = 2e-4
```

---

# LoRA Configuration

```python
r = 16

lora_alpha = 16

target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

---

# Docker Deployment

## Build Docker Image

```bash
docker build -t inferra-qwen .
```

---

# Run Docker Container

```bash
docker run --gpus all -p 8000:8000 inferra-qwen
```

---

# Dockerfile

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY . .

RUN pip install -U \
    transformers \
    accelerate \
    peft \
    bitsandbytes \
    unsloth \
    fastapi \
    uvicorn

EXPOSE 8000

CMD ["python", "app.py"]
```

---

# FastAPI Example

```python
from fastapi import FastAPI
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

app = FastAPI()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
)

model = PeftModel.from_pretrained(
    model,
    "mohdmusheer/inferra",
)

FastLanguageModel.for_inference(model)

@app.get("/")
def home():
    return {
        "message": "Inferra Model Running"
    }
```

---

# Docker Hub Description

```text
Inferra-Qwen-LoRA

Lightweight conversational AI model built using QLoRA fine-tuning on Qwen2.5-3B-Instruct.

Features:
- 4-bit Quantized Inference
- LoRA Adapters
- FastAPI Deployment
- Hugging Face Compatible
- Consumer GPU Friendly

Frameworks:
Transformers, PEFT, Unsloth

GPU:
NVIDIA T4
```

---

# Docker Quick Start

```bash
docker pull YOUR_USERNAME/inferra-qwen

docker run --gpus all -p 8000:8000 YOUR_USERNAME/inferra-qwen
```

---

# Hugging Face Tags

```text
qwen
qlora
peft
unsloth
transformers
4bit
quantized
chatbot
instruction-tuning
conversational
```

---

# Project Structure

```text
├── app.py
├── Dockerfile
├── inference.py
├── training.ipynb
├── requirements.txt
└── README.md
```

---

# Limitations

This model is:
- Not a full fine-tuned foundation model
- Not a frontier reasoning model
- Not deeply domain-specialized

The adapter primarily modifies:
- Conversational behavior
- Response formatting
- Instruction-following style

Most intelligence still comes from the base Qwen model.

---

# Future Improvements

Potential future upgrades:
- Reasoning datasets
- Coding specialization
- RAG integration
- vLLM deployment
- GGUF export
- Ollama support
- Synthetic dataset generation
- Evaluation benchmarks

---

# Performance Notes

Performance depends on:
- Prompt quality
- Context length
- GPU hardware
- Quantization settings

Longer context windows significantly increase:
- VRAM usage
- Attention cost
- Inference latency

---

# Acknowledgements

Built using:
- Unsloth
- Hugging Face Transformers
- PEFT
- BitsAndBytes
- Qwen Models

---

# License

This project follows the license terms of:
- Base model
- Training datasets
- Hugging Face ecosystem

Please review respective licenses before commercial usage.