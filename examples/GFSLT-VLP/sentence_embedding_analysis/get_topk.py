import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-german-cased")
model = BertModel.from_pretrained("google-bert/bert-base-german-cased")

# Function to compute sentence embeddings
def get_sentence_embedding(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=50)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()

# Load sentences from file
# file_path = "./out/GF_SignCL_viz_topk/tmp_refs.txt" 
file_path = "./out/GF_SignCL_viz_topk/tmp_refs.txt"  

with open(file_path, "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in file.readlines() if line.strip()]

# Compute embeddings for all sentences
embeddings = np.array([get_sentence_embedding(sentence) for sentence in sentences])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Find top-5 most similar sentences for each sentence
top_n = 5
results = []

for i, sentence in enumerate(sentences):
    # Get similarity scores for the sentence
    sim_scores = similarity_matrix[i]
    
    # Sort and get top-5 most similar sentences (excluding itself)
    top_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
    
    # Store results
    similar_sentences = [sentences[idx] for idx in top_indices]
    results.append([sentence] + similar_sentences)

# Function to append similarity scores
def append_similarity_scores(df, similarity_matrix, top_n):
    for i in range(top_n):
        df[f"Top-{i+1} Similarity"] = [similarity_matrix[idx, np.argsort(similarity_matrix[idx])[::-1][1:top_n+1][i]] for idx in range(len(df))]
    return df

# Convert results to DataFrame
df = pd.DataFrame(results, columns=["Sentence", "Top-1 Similar", "Top-2 Similar", "Top-3 Similar", "Top-4 Similar", "Top-5 Similar"])

# Append similarity scores
df = append_similarity_scores(df, similarity_matrix, top_n)

# Save to file
output_file = "./top5_similar_sentences.txt"
df.to_csv(output_file, sep="\t", index=False)

print(f"Results saved to {output_file}")
