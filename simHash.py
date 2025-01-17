import hashlib
import numpy as np
import pandas as pd

# Function to preprocess the JSON structure into a comparable string
def json_to_string(element):
    attributes = []
    for key, value in element.items():
        attributes.append(f"{key}:{value}")
    return " ".join(attributes)

# Function to calculate the SimHash of a text
def simhash(text, hash_size=64):
    tokens = text.split()
    vectors = np.zeros(hash_size, dtype=int)

    for token in tokens:
        hash_value = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
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

# Load JSON data
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

# Extract elements from both metamodels
scrum_elements = list(scrum_json["Scrum"].items())[:5]
prince2_elements = list(prince2_json["PRINCE2"].items())[:5]

# Prepare results and comparison table
simhash_table = np.zeros((5, 5), dtype=int)
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
threshold = 10  # SimHash distance threshold for a match

# Compare elements
for i, (scrum_key, scrum_value) in enumerate(scrum_elements):
    scrum_string = json_to_string(scrum_value)
    scrum_hash = simhash(scrum_string)

    for j, (prince2_key, prince2_value) in enumerate(prince2_elements):
        prince2_string = json_to_string(prince2_value)
        prince2_hash = simhash(prince2_string)

        # Calculate SimHash distance
        simhash_distance = hamming_distance(scrum_hash, prince2_hash)
        simhash_table[i][j] = simhash_distance

        # Determine predicted correspondence
        predicted_correspondence = 1 if simhash_distance <= threshold else 0
        true_correspondence = 1 if i == j else 0  # Diagonal is the ground truth

        # Calculate metrics
        if predicted_correspondence == 1 and true_correspondence == 1:
            true_positives += 1
        elif predicted_correspondence == 1 and true_correspondence == 0:
            false_positives += 1
        elif predicted_correspondence == 0 and true_correspondence == 1:
            false_negatives += 1
        else:
            true_negatives += 1

# Calculate Precision, Recall, and F1-score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Convert SimHash table to DataFrame for readability
simhash_df = pd.DataFrame(simhash_table, columns=[key for key, _ in prince2_elements],
                          index=[key for key, _ in scrum_elements])

# Print results
print("SimHash Distance Table (5x5):\n")
print(simhash_df)

print("\nMetrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
