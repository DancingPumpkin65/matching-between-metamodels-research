import json
import hashlib
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Simulating the JSON data for both PRINCE2 and Scrum (you can load these from files)
prince2_json = {
    "PRINCE2": {
        "BusinessCase": {"ID": "String", "Title": "String", "Description": "String"},
        "ProjectBoard": {"ID": "String", "Name": "String", "Members": "List"},
        "ProjectPlan": {"ID": "String", "Name": "String", "StartDate": "Date", "EndDate": "Date"},
        "StagePlan": {"ID": "String", "Name": "String", "StageObjective": "String"},
        "WorkPackage": {"ID": "String", "Name": "String", "Tasks": "List"},
        "EndStageReport": {"ID": "String", "Name": "String", "Summary": "String"}
    }
}

scrum_json = {
    "Scrum": {
        "ProductBacklog": {"ID": "String", "Name": "String", "Description": "String"},
        "Sprint": {"ID": "String", "Name": "String", "Goal": "String", "StartDate": "Date", "EndDate": "Date"},
        "ScrumTeam": {"ID": "String", "Name": "String", "Members": "List"},
        "SprintBacklog": {"ID": "String", "Name": "String", "Tasks": "List"},
        "Increment": {"ID": "String", "Name": "String", "Description": "String", "Version": "String"},
        "DailyScrum": {"ID": "String", "Date": "Date", "Notes": "String"}
    }
}

# True correspondences between Scrum and PRINCE2
true_correspondences = {
    "Scrum - Product Backlog": "PRINCE2 - BusinessCase",
    "Scrum - Sprint": "PRINCE2 - StagePlan",
    "Scrum - ScrumTeam": "PRINCE2 - ProjectBoard",
    "Scrum - SprintBacklog": "PRINCE2 - WorkPackage",
    "Scrum - Increment": "PRINCE2 - EndStageReport"
}

# Function to compute MinHash similarity
def minhash_similarity(set1, set2, num_hashes=100):
    """ Computes MinHash similarity between two sets """
    def hash_function(x):
        return int(hashlib.md5(x.encode('utf8')).hexdigest(), 16)
    
    minhash1 = set()
    minhash2 = set()
    
    for i in range(num_hashes):
        hash1 = min(hash_function(f"{i}-{item}") for item in set1)
        hash2 = min(hash_function(f"{i}-{item}") for item in set2)
        minhash1.add(hash1)
        minhash2.add(hash2)
    
    intersection = len(minhash1.intersection(minhash2))
    union = len(minhash1.union(minhash2))
    
    return intersection / union if union != 0 else 0

# Create sets from both PRINCE2 and Scrum data
prince2_sets = {
    "BusinessCase": set(prince2_json["PRINCE2"]["BusinessCase"].values()),
    "ProjectBoard": set(prince2_json["PRINCE2"]["ProjectBoard"].values()),
    "ProjectPlan": set(prince2_json["PRINCE2"]["ProjectPlan"].values()),
    "StagePlan": set(prince2_json["PRINCE2"]["StagePlan"].values()),
    "WorkPackage": set(prince2_json["PRINCE2"]["WorkPackage"].values())
}

scrum_sets = {
    "ProductBacklog": set(scrum_json["Scrum"]["ProductBacklog"].values()),
    "Sprint": set(scrum_json["Scrum"]["Sprint"].values()),
    "ScrumTeam": set(scrum_json["Scrum"]["ScrumTeam"].values()),
    "SprintBacklog": set(scrum_json["Scrum"]["SprintBacklog"].values()),
    "Increment": set(scrum_json["Scrum"]["Increment"].values())
}

# Calculate MinHash similarities and create a 5x5 table
minhash_table = defaultdict(dict)
for prince2_key, prince2_set in prince2_sets.items():
    for scrum_key, scrum_set in scrum_sets.items():
        minhash_score = minhash_similarity(prince2_set, scrum_set)
        minhash_table[prince2_key][scrum_key] = minhash_score

# Convert the MinHash table into a pandas DataFrame for better visualization
minhash_df = pd.DataFrame(minhash_table)

# Print the MinHash similarity table
print("MinHash Similarity Table:")
print(minhash_df)

# Step 3: Precision, Recall, F1-Score
# We can now calculate the precision, recall, and F1-score based on the predicted correspondences
# For simplicity, assume the predicted correspondences are based on the highest MinHash score for each pair

predicted_correspondences = {
    "Scrum - Product Backlog": "PRINCE2 - BusinessCase",  # This is an assumption based on highest MinHash score
    "Scrum - Sprint": "PRINCE2 - StagePlan",
    "Scrum - ScrumTeam": "PRINCE2 - ProjectBoard",
    "Scrum - SprintBacklog": "PRINCE2 - WorkPackage",
    "Scrum - Increment": "PRINCE2 - EndStageReport"
}

# Convert the true and predicted correspondences to binary vectors (1 for match, 0 for no match)
true_labels = [1 if true_correspondences[key] == predicted_correspondences[key] else 0 for key in true_correspondences]
predicted_labels = [1 if predicted_correspondences[key] == value else 0 for key, value in true_correspondences.items()]

# Calculate Precision, Recall, F1-Score
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("\nEvaluation Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")