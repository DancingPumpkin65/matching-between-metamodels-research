from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Define the elements of the two metamodels
metamodel_scrum = ["Product Backlog", "Sprint", "Scrum Master", "Daily Scrum", "Increment"]
metamodel_prince2 = ["Product Description", "Stage Plan", "Project Manager", "Daily Log", "Project Deliverable"]

# Define true correspondences
true_correspondences = {
    "Product Backlog": "Product Description",
    "Sprint": "Stage Plan",
    "Scrum Master": "Project Manager",
    "Daily Scrum": "Daily Log",
    "Increment": "Project Deliverable"
}

# Load the pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for both metamodels
embeddings_scrum = model.encode(metamodel_scrum, convert_to_tensor=True)
embeddings_prince2 = model.encode(metamodel_prince2, convert_to_tensor=True)

# Calculate cosine similarity
similarity_matrix = util.cos_sim(embeddings_scrum, embeddings_prince2)

# Convert similarity matrix to a DataFrame for better visualization
similarity_df = pd.DataFrame(
    similarity_matrix.numpy(),
    index=metamodel_scrum,
    columns=metamodel_prince2
)

# Display the similarity table
print("5x5 Table of SBERT Similarities:")
print(similarity_df)

# Set a similarity threshold to determine predicted correspondences
threshold = 0.5  # Adjust as needed
predicted_correspondences = {}

# Identify predicted correspondences
for i, row in similarity_df.iterrows():
    best_match = row.idxmax()  # Find the column with the highest similarity
    if row[best_match] >= threshold:
        predicted_correspondences[i] = best_match

# Calculate Precision, Recall, and F1-score
true_positive = sum(1 for k, v in predicted_correspondences.items() if true_correspondences.get(k) == v)
false_positive = len(predicted_correspondences) - true_positive
false_negative = len(true_correspondences) - true_positive

precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Display the results
print("\nEvaluation Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
