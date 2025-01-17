import hashlib
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Preprocessing function for text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stopwords.words('english') and word not in string.punctuation
    ]
    return tokens

# Function to calculate the hash of a token
def token_hash(token, hash_size=64):
    return int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16)

# Enhanced SimHash function with preprocessing
def enhanced_simhash(text, hash_size=64):
    tokens = preprocess_text(text)
    vectors = np.zeros(hash_size, dtype=int)

    for token in tokens:
        hash_value = token_hash(token)
        for i in range(hash_size):
            bit = (hash_value >> i) & 1
            vectors[i] += 1 if bit else -1

    simhash_value = 0
    for i in range(hash_size):
        if vectors[i] > 0:
            simhash_value |= (1 << i)

    return simhash_value

# Function to calculate the Hamming distance between two SimHashes
def hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count('1')

# Example data (your actual dataset should follow this format)
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

# Initialize metrics and results
results = []
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Prepare a table to store the SimHash distances for display
simhash_table = np.zeros((5, 5), dtype=int)

# Loop through each element in Métamodèle 1
for i, row1 in df.iterrows():
    for j, row2 in df.iterrows():
        # Calculate the SimHash for both descriptions
        simhash1 = enhanced_simhash(row1['Description Élément 1'])
        simhash2 = enhanced_simhash(row2['Description Élément 2'])
        
        # Calculate the Hamming distance
        simhash_distance = hamming_distance(simhash1, simhash2)
        
        # Store the SimHash distance in the table
        simhash_table[i][j] = simhash_distance

        # Get the predicted Correspondance value
        predicted_correspondance = 1 if simhash_distance <= 10 else 0  # Threshold can be tuned
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
            "SimHash Distance": simhash_distance,
            "Predicted Correspondance": predicted_correspondance,
            "True Correspondance": true_correspondance
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Calculate Precision, Recall, and F1-score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print the SimHash distance table and metrics
print("SimHash Distance Table (5x5):\n")
simhash_df = pd.DataFrame(simhash_table, columns=df["Élément 2"], index=df["Élément 1"])
print(simhash_df)

# Print the metrics
print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
