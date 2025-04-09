import pandas as pd
from collections import Counter

# Load the PHOENIX-2014-T dataset
csv_file_path = "./data/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv"
df_csv = pd.read_csv(csv_file_path, delimiter="|")

# Load the top-k similar sentences file
top_k_file_path = "./top5_similar_sentences.txt"
df_top_k = pd.read_csv(top_k_file_path, sep="\t")

# Merge on the 'translation' column, assuming sentences match here
merged_df = df_top_k.merge(df_csv, left_on="Sentence", right_on="translation", how="left")

# Create a dictionary mapping `translation` to `orth` for quick lookup
translation_to_orth = dict(zip(df_csv["translation"], df_csv["orth"]))

# Add `orth` for each top-K similar sentence
for i in range(1, 6):  # Top-1 to Top-5 Similar
    merged_df[f"Top-{i} Orth"] = merged_df[f"Top-{i} Similar"].map(translation_to_orth)

# Function to compute word overlap between two orth sentences
def compute_word_overlap(orth1, orth2):
    if pd.notna(orth1) and pd.notna(orth2):
        # print(orth1 + "|||" + orth2)
        words1 = set(orth1.split())
        words2 = set(orth2.split())
        return len(words1 & words2)  # Count of shared words
    return 0
merged_df["Gloss_length"] = merged_df["orth"].apply(
    lambda row: len(str(row).split()) if pd.notna(row) else 0
)
# print(merged_df['Gloss_length'].head())
# Compute word overlap between target sentence's orth and each top-K orth
for i in range(1, 6):
    merged_df[f"Word_Overlap_Top_{i}"] = merged_df.apply(
        lambda row: compute_word_overlap(row["orth"], row[f"Top-{i} Orth"]), axis=1
    )

# # Save the final merged file
output_final_file = "./merged_top5_orth_word_overlap_v2.csv"
merged_df.to_csv(output_final_file, sep=",", index=False)
