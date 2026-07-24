# LLM Fine-Tuning for Structured Document Extraction

Fine-tuned **Phi-3-mini** language model using **LoRA/PEFT** to extract structured JSON from semi-structured HTML documents. The project demonstrates an end-to-end supervised fine-tuning workflow including dataset preparation, model training, evaluation on a held-out test set, and deployment using GGUF for lightweight local inference.

---

## Project Overview

This project focuses on adapting an open-weight Large Language Model (LLM) to perform structured document extraction.

Given an HTML document, the fine-tuned model generates structured JSON containing the required information.

Example:

### Input

```html
<div class="product">
    <h2>Wireless Mouse</h2>
    <p>Price: $25</p>
    <p>Brand: Logitech</p>
</div>
```

### Output

```json
{
  "Product": "Wireless Mouse",
  "Price": "$25",
  "Brand": "Logitech"
}
```

---

# Features

- Fine-tuned Phi-3-mini using LoRA/PEFT
- Supervised Fine-Tuning (SFT) with TRL SFTTrainer
- Prompt-based HTML → JSON extraction
- Train/Test split for evaluation
- Mixed Precision Training
- Gradient Checkpointing
- GGUF model export
- Local deployment using llama.cpp and Ollama

---

# Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Unsloth
- PEFT (LoRA)
- TRL
- Datasets
- llama.cpp
- Ollama

---

# Dataset

The dataset contains HTML snippets paired with their corresponding structured JSON outputs.

Example structure:

```json
{
  "input": "<html>...</html>",
  "output": {
      "Product":"...",
      "Price":"...",
      "Brand":"..."
  }
}
```

The dataset was split into:

- 85% Training
- 15% Held-out Test

using

```python
train_test_split(test_size=0.15, random_state=42)
```

---

# Model

Base Model

- Phi-3-mini (4-bit)

Fine-Tuning

- LoRA / PEFT
- Rank (r): 64
- LoRA Alpha: 128

Target Modules

- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

Only **119,537,664 trainable parameters (3.03%)** were updated during fine-tuning.

---

# Training Configuration

- Epochs: 3
- Learning Rate: 2e-4
- Batch Size: 2
- Gradient Accumulation: 4
- Optimizer: AdamW 8-bit
- Mixed Precision Training
- Gradient Checkpointing
- Sequence Length: 2048

---

# Evaluation

The model was evaluated on a held-out test split that was not used during training.

Metrics:

- Exact Match Accuracy
- Field Accuracy

Results

| Metric | Baseline | Fine-tuned |
|---------|----------|------------|
| Exact Match | 0% | 100% |
| Field Accuracy | 0% | 100% |

---

# Deployment

After fine-tuning,

the LoRA adapters were merged into the base model and exported in **GGUF (Q4_K_M)** format.

The exported model can be used with:

- llama.cpp
- Ollama

Example

```bash
ollama create phi3-json -f Modelfile
```

---

# Repository Structure

```
├── dataset/
├── notebooks/
│   └── Fine_Tuning_Project.ipynb
├── gguf_model/
├── outputs/
├── README.md
```

---

# Learning Outcomes

Through this project I learned:

- Parameter-Efficient Fine-Tuning (LoRA)
- Supervised Fine-Tuning of LLMs
- Prompt formatting for instruction tuning
- Dataset preparation using Hugging Face Datasets
- Training using TRL SFTTrainer
- Model evaluation on unseen data
- Model quantization and GGUF export
- Local inference using Ollama

---

# Future Improvements

- Evaluate on larger and more diverse document datasets
- Experiment with larger instruction-tuned models
- Support multiple document templates
- Build a web interface for document upload and extraction

---

# Acknowledgements

- Microsoft Phi-3
- Hugging Face
- Unsloth
- TRL
- PEFT
