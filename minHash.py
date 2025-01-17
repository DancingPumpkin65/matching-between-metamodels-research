import numpy as np
from datasketch import MinHash
from sklearn.metrics import precision_recall_fscore_support

# Step 1: Define Scrum and PRINCE2 elements
scrum_elements = ["Product Backlog", "Sprint", "Sprint Backlog", "Daily Scrum", "Sprint Review"]
prince2_elements = ["Product Description", "Stage Plan", "Stage Boundary", "Daily Standup", "End Stage Report"]

# Step 2: Define True Correspondences
true_correspondences = {
    "Product Backlog": "Product Description",
    "Sprint": "Stage Plan",
    "Sprint Backlog": "Stage Boundary",
    "Daily Scrum": "Daily Standup",
    "Sprint Review": "End Stage Report"
}

# Step 3: Function to compute MinHash similarity
def compute_minhash_similarity(set1, set2, num_hashes=200):
    minhash1 = MinHash()
    minhash2 = MinHash()
    
    for element in set1:
        minhash1.update(element.encode('utf8'))
    for element in set2:
        minhash2.update(element.encode('utf8'))
    
    return minhash1.jaccard(minhash2)

# Step 4: Compute the MinHash similarities for all pairs
similarities = np.zeros((len(scrum_elements), len(prince2_elements)))

for i, scrum in enumerate(scrum_elements):
    for j, prince2 in enumerate(prince2_elements):
        similarities[i, j] = compute_minhash_similarity(set(scrum.split()), set(prince2.split()))

# Step 5: Show the similarity table (5x5)
print("MinHash Similarity Table (5x5):")
for i, scrum in enumerate(scrum_elements):
    for j, prince2 in enumerate(prince2_elements):
        print(f"{scrum} <-> {prince2}: {similarities[i, j]:.4f}")

# Step 6: Predicted correspondences based on MinHash similarities
# Let's assume the predicted correspondences are based on the highest similarity score
predicted_correspondences = {}
for i, scrum in enumerate(scrum_elements):
    best_match_idx = np.argmax(similarities[i])
    predicted_correspondences[scrum] = prince2_elements[best_match_idx]

# Step 7: Calculate Precision, Recall, F1-Score
y_true = []
y_pred = []

# Create binary vectors for true and predicted correspondences
for scrum in scrum_elements:
    for prince2 in prince2_elements:
        if true_correspondences.get(scrum) == prince2:
            y_true.append(1)
        else:
            y_true.append(0)
        if predicted_correspondences.get(scrum) == prince2:
            y_pred.append(1)
        else:
            y_pred.append(0)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

# Step 8: Display Precision, Recall, and F1-Score
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")