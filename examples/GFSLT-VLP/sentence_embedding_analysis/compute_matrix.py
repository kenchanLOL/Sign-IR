import pickle
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-german-cased")
model = BertModel.from_pretrained("google-bert/bert-base-german-cased")

# Function to compute sentence embeddings
def get_sentence_embeddings_batch(sentences, batch_size=32):
    embeddings = []
    dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)
    for batch in tqdm(dataloader, desc="Computing embeddings in batches"):
        tokens = tokenizer(list(batch), return_tensors="pt", truncation=True, padding="max_length", max_length=50)
        with torch.no_grad():
            output = model(**tokens)
        batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def compute_word_overlap(orth1, orth2):
    if orth1 and orth2:
        words1 = set(orth1)
        words2 = set(orth2)
        return len(words1 & words2)  / len(words1) # Count of shared words
    return 0

# Load sentences from file
# file_path = "./out/GF_SignCL_viz_topk/tmp_refs.txt" 
file_path = "/home/chan0305/sign_lang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train-complex-annotation.corpus.csv"  


with open(file_path, "r", encoding="utf-8") as file:
    sentences = []
    glosses = []
    for line in file.readlines():
        if line.strip():
            row = line.strip().split("|")
            sentences.append(row[-1])
            glosses.append(row[-2].split())
    sentences = sentences[1:]
    glosses = glosses[1:]
    # sentences = [line.split("|")[-1].strip() for line in file.readlines() if line.strip()]
print(f"Total number of sentence in dataset : {len(sentences)}")
# Compute embeddings for all sentences
embeddings = get_sentence_embeddings_batch(sentences, batch_size=32)

# # Compute cosine similarity matrix
# similarity_matrix = cosine_similarity(embeddings)

# # Compute word overlap matrix using numpy broadcasting
# # glosses_array = np.array(glosses, dtype=object)
# word_overlap_matrix = np.zeros((len(glosses), len(glosses)))

# for i in tqdm(range(len(glosses)), desc="Computing word overlap matrix"):
#     word_overlap_matrix[i, :] = [
#         compute_word_overlap(glosses[i], glosses[j]) for j in range(len(glosses))
#     ]

# print("Word overlap matrix computed.")
with open("embeddings.pkl", "wb") as emb_file:
    pickle.dump(embeddings, emb_file)
# Save matrices to pickle files
# with open("similarity_matrix.pkl", "wb") as sim_file:
#     pickle.dump(similarity_matrix, sim_file)

# with open("word_overlap_matrix.pkl", "wb") as overlap_file:
#     pickle.dump(word_overlap_matrix, overlap_file)

# print("Matrices saved to pickle files.")
