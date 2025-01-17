import hashlib
import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to tokenize and preprocess text
def tokenize(description):
    # Remove punctuation and lowercase
    tokens = re.findall(r'\w+', description.lower())
    # Apply stemming
    return set(ps.stem(token) for token in tokens)

# Function to hash a set of tokens using different hash functions
def minhash(tokens, num_hashes=200):
    min_hashes = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for token in tokens:
            hash_value = int(hashlib.sha256((str(i) + token).encode('utf-8')).hexdigest(), 16)
            min_hash = min(min_hash, hash_value)
        min_hashes.append(min_hash)
    return min_hashes

# Function to calculate Jaccard similarity between two sets based on MinHash signatures
def minhash_similarity(set1, set2, num_hashes=200):
    minhash1 = minhash(set1, num_hashes)
    minhash2 = minhash(set2, num_hashes)
    
    # Calculate the number of hash functions that give the same result for both sets
    matches = sum([1 for i in range(num_hashes) if minhash1[i] == minhash2[i]])
    
    # Approximate the Jaccard similarity based on the fraction of hash functions that match
    return matches / num_hashes

# Example data
data = {
    "Métamodèle 1": ["Scrum", "Scrum", "Scrum", "Scrum", "Scrum"],
    "Élément 1": ["Product Backlog", "Sprint", "User Stories", "Sprint Review", "Sprint Retrospective"],
    "Description Élément 1": [
        "A prioritized list of work for the team",
        "A time-boxed iteration of work",
        "Short, simple descriptions of a feature",
        "A meeting to review the sprint work",
        "A meeting to reflect on the sprint"
    ],
    "Métamodèle 2": ["PRINCE2", "PRINCE2", "PRINCE2", "PRINCE2", "PRINCE2"],
    "Élément 2": ["Product Description", "Stage Plan", "Work Packages", "Checkpoint Report", "End Stage Report"],
    "Description Élément 2": [
        "A detailed description of a product",
        "A detailed plan for a specific stage",
        "A group of related tasks",
        "A report of the progress at a checkpoint",
        "A report summarizing the end of a stage"
    ],
    "Correspondance": [1, 1, 1, 1, 1]  # True matching pairs
}

# Create DataFrame
df = pd.DataFrame(data)

# Initialize an empty list to store results
results = []
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Prepare a table to store the MinHash similarities for display
minhash_table = np.zeros((5, 5), dtype=float)

# Loop through each element in Métamodèle 1
for i in range(len(df)):
    for j in range(i, len(df)):  # Avoid redundant comparisons
        row1 = df.iloc[i]
        row2 = df.iloc[j]

        # Tokenize the descriptions for MinHash
        set1 = tokenize(row1['Description Élément 1'])
        set2 = tokenize(row2['Description Élément 2'])
        
        # Calculate the MinHash similarity
        minhash_similarity_value = minhash_similarity(set1, set2, num_hashes=200)
        
        # Store the MinHash similarity in the table
        minhash_table[i][j] = minhash_similarity_value
        minhash_table[j][i] = minhash_similarity_value  # Symmetric matrix

        # Get the predicted Correspondance value
        predicted_correspondance = 1 if minhash_similarity_value >= 0.5 else 0  # Thresholding to predict 1 or 0
        true_correspondance = 1 if i == j else 0  # Only the diagonal should be 1 (the true matching pairs)

        # Count True Positives, False Positives, True Negatives, and False Negatives
        if predicted_correspondance == 1 and true_correspondance == 1:
            true_positives += 1
        elif predicted_correspondance == 1 and true_correspondance == 0:
            false_positives += 1
        elif predicted_correspondance == 0 and true_correspondance == 1:
            false_negatives += 1
        else:
            true_negatives += 1

        # Store the result for this comparison
        results.append({
            "Métamodèle 1": row1['Métamodèle 1'],
            "Élément 1": row1['Élément 1'],
            "Métamodèle 2": row2['Métamodèle 2'],
            "Élément 2": row2['Élément 2'],
            "MinHash Similarity": minhash_similarity_value,
            "Predicted Correspondance": predicted_correspondance,
            "True Correspondance": true_correspondance
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Calculate Precision, Recall, and F1-score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print the MinHash similarity table and metrics
print("MinHash Similarity Table (5x5):\n")
minhash_df = pd.DataFrame(minhash_table, columns=df["Élément 2"], index=df["Élément 1"])
print(minhash_df)

# Print the metrics
print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
