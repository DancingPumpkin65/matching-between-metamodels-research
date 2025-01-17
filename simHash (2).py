from simhash import Simhash
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

# Sample data for the Scrum and PRINCE2 elements
scrum_elements = [
    "Product Backlog",
    "Sprint",
    "Daily Scrum",
    "Sprint Review",
    "Sprint Retrospective"
]

prince2_elements = [
    "Product Description",
    "Stage Plan",
    "Daily Standup",
    "End Stage Assessment",
    "Lessons Learned"
]

# Define true correspondences (based on your instruction)
true_correspondences = {
    "Product Backlog": "Product Description",
    "Sprint": "Stage Plan",
    "Daily Scrum": "Daily Standup",
    "Sprint Review": "End Stage Assessment",
    "Sprint Retrospective": "Lessons Learned"
}

# Function to compute SimHash similarity
def simhash_similarity(str1, str2):
    return Simhash(str1).distance(Simhash(str2))

# Compute similarity table
def compute_similarity_table(scrum_elements, prince2_elements):
    similarity_table = pd.DataFrame(np.zeros((len(scrum_elements), len(prince2_elements))), columns=prince2_elements, index=scrum_elements)
    
    for i, scrum_item in enumerate(scrum_elements):
        for j, prince_item in enumerate(prince2_elements):
            similarity_table.iloc[i, j] = simhash_similarity(scrum_item, prince_item)
    
    return similarity_table

# Generate the similarity table
similarity_table = compute_similarity_table(scrum_elements, prince2_elements)
print("Similarity Table:")
print(similarity_table)

# Predict correspondences (based on similarity, here we choose the closest match)
predicted_correspondences = {}
for scrum_item in scrum_elements:
    closest_match = prince2_elements[np.argmin(similarity_table.loc[scrum_item].values)]  # Choose the one with min distance
    predicted_correspondences[scrum_item] = closest_match

print("\nPredicted Correspondences:")
for scrum_item, predicted_item in predicted_correspondences.items():
    print(f"{scrum_item} -> {predicted_item}")

# Compare predicted correspondences with true correspondences
true_labels = [true_correspondences[scrum_item] for scrum_item in scrum_elements]
predicted_labels = [predicted_correspondences[scrum_item] for scrum_item in scrum_elements]

# Calculate precision, recall, and F1-score
precision = precision_score(true_labels, predicted_labels, average='micro', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='micro', zero_division=1)
f1 = f1_score(true_labels, predicted_labels, average='micro', zero_division=1)

print("\nMetrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
