import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# -----------------------------
# Load ClinicalBERT
# -----------------------------

def load_clinicalbert():

    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )

    model = AutoModel.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )

    model.eval()

    return tokenizer, model


# -----------------------------
# Generate BERT embedding
# -----------------------------

def get_bert_embedding(text, tokenizer, model, max_length=256):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    with torch.no_grad():

        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.squeeze().numpy()


# -----------------------------
# Batch embedding generation
# -----------------------------

def build_bert_embeddings(texts, tokenizer, model):

    embeddings = []

    for text in tqdm(texts):

        emb = get_bert_embedding(text, tokenizer, model)

        embeddings.append(emb)

    return np.vstack(embeddings)
