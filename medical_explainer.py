import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ===================================================
# Hugging Face Token (SAFE WAY)
# ===================================================

# Correct: Read token from environment variable name
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    print("⚠️ Warning: HF_TOKEN not found. Using anonymous access (slower downloads).")

# Avoid tokenizer parallelism warning on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------
# Load ClinicalBERT
# -------------------------------
clinical_tokenizer = AutoTokenizer.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    token=HF_TOKEN
)

clinical_model = AutoModel.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    token=HF_TOKEN
)

# -------------------------------
# Load BioGPT
# -------------------------------
biogpt_tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/biogpt",
    token=HF_TOKEN,
    use_fast=False
)

biogpt_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/biogpt",
    token=HF_TOKEN
)


def generate_medical_explanation(term, context_text=""):

    combined_text = f"Medical term: {term}. Context: {context_text}"

    inputs = clinical_tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        _ = clinical_model(**inputs)

    prompt = (
        f"Explain the medical term '{term}' in simple language "
        f"so that a patient can understand it.\nExplanation:"
    )

    gen_inputs = biogpt_tokenizer(prompt, return_tensors="pt")

    outputs = biogpt_model.generate(
        **gen_inputs,
        max_length=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    explanation = biogpt_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return explanation.strip()
