import hashlib
import re
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to tokenize and preprocess text
def tokenize(text):
    # Remove punctuation and lowercase
    tokens = re.findall(r'\w+', text.lower())
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

# Example data (scrum_json and prince2_json)
scrum_json = {
    "Scrum": {
        "ProductBacklog": {"ID": "String", "Name": "String", "Description": "String"},
        "Sprint": {"ID": "String", "Name": "String", "Goal": "String", "StartDate": "Date", "EndDate": "Date"},
        "ScrumTeam": {"ID": "String", "Name": "String", "Members": "List"},
        "SprintBacklog": {"ID": "String", "Name": "String", "Tasks": "List"},
        "Increment": {"ID": "String", "Name": "String", "Description": "String", "Version": "String"}
    }
}

prince2_json = {
    "PRINCE2": {
        "BusinessCase": {"ID": "String", "Title": "String", "Description": "String"},
        "ProjectBoard": {"ID": "String", "Name": "String", "Members": "List"},
        "ProjectPlan": {"ID": "String", "Name": "String", "StartDate": "Date", "EndDate": "Date"},
        "StagePlan": {"ID": "String", "Name": "String", "StageObjective": "String"},
        "WorkPackage": {"ID": "String", "Name": "String", "Tasks": "List"}
    }
}

# Extract descriptions for comparison from both Scrum and PRINCE2
scrum_descriptions = [
    scrum_json['Scrum']['ProductBacklog']['Description'],
    scrum_json['Scrum']['Sprint']['Goal'],
    scrum_json['Scrum']['Increment']['Description']
]

prince2_descriptions = [
    prince2_json['PRINCE2']['BusinessCase']['Description'],
    prince2_json['PRINCE2']['StagePlan']['StageObjective']
]

# Initialize an empty list to store similarity results
similarity_results = []
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Calculate MinHash similarity between each Scrum and PRINCE2 description
for i, scrum_desc in enumerate(scrum_descriptions):
    for j, prince2_desc in enumerate(prince2_descriptions):
        # Tokenize the descriptions
        scrum_set = tokenize(scrum_desc)
        prince2_set = tokenize(prince2_desc)
        
        # Calculate the MinHash similarity
        similarity_value = minhash_similarity(scrum_set, prince2_set, num_hashes=200)
        
        # Store the result in the similarity table
        similarity_results.append({
            "Scrum Element": list(scrum_json['Scrum'].keys())[i],
            "PRINCE2 Element": list(prince2_json['PRINCE2'].keys())[j],
            "MinHash Similarity": similarity_value
        })

        # Predict correspondance (thresholding at 0.5)
        predicted_correspondance = 1 if similarity_value >= 0.5 else 0
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

# Convert similarity results to a DataFrame
similarity_df = pd.DataFrame(similarity_results)

# Convert the similarity values into a 5x5 table (using NaN for non-existent cells)
minhash_table = np.zeros((5, 5), dtype=float)
for i, scrum_desc in enumerate(scrum_descriptions):
    for j, prince2_desc in enumerate(prince2_descriptions):
        minhash_table[i][j] = minhash_similarity(tokenize(scrum_desc), tokenize(prince2_desc), num_hashes=200)

# Calculate Precision, Recall, and F1-score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print the MinHash similarity table (5x5) and metrics
print("MinHash Similarity Table (5x5):\n")
minhash_df = pd.DataFrame(minhash_table, columns=list(prince2_json['PRINCE2'].keys()), index=list(scrum_json['Scrum'].keys()))
print(minhash_df)

# Print the metrics
print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
