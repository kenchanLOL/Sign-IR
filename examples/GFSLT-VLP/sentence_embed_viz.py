import torch
from transformers import BertModel, BertTokenizer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-german-cased")
model = BertModel.from_pretrained("google-bert/bert-base-german-cased")

# Function to compute sentence embeddings
def get_sentence_embedding(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=50)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()
file_path = "./out/GF_SignCL_viz_topk/tmp_refs.txt"
# Compute embeddings for all sentences
with open(file_path, "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in file.readlines() if line.strip()]
embeddings = np.array([get_sentence_embedding(sentence) for sentence in sentences])

# Perform t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Create scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
plt.title("t-SNE Visualization of Weather Sentences using BERT Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

# Save the plot
plot_path = "./tsne_weather_sentences.png"
plt.savefig(plot_path)
plt.show()

# Return the plot file path for download
# plot_path