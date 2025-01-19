import hashlib

# Function to calculate the Hamming distance between two SimHashes
def hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count('1')

# Function to calculate the SimHash of a string
def simhash(value):
    hash_value = int(hashlib.md5(value.encode('utf-8')).hexdigest(), 16)
    return hash_value

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
        "A detailed plan for a stage",
        "A set of work packages",
        "A report on the progress of work",
        "A report on the completion of a stage"
    ]
}

# Calculate SimHashes for each description
simhashes_1 = [simhash(desc) for desc in data["Description Élément 1"]]
simhashes_2 = [simhash(desc) for desc in data["Description Élément 2"]]

# Calculate Hamming distances and store in a 5x5 table
distance_table = [[hamming_distance(h1, h2) for h2 in simhashes_2] for h1 in simhashes_1]

# Define true correspondences
true_correspondences = {
    ("Product Backlog", "Product Description"),
    ("Sprint", "Stage Plan"),
    ("User Stories", "Work Packages"),
    ("Sprint Review", "Checkpoint Report"),
    ("Sprint Retrospective", "End Stage Report")
}

# Determine predicted correspondences based on minimum distance
predicted_correspondences = set()
for i, row in enumerate(distance_table):
    min_distance = min(row)
    j = row.index(min_distance)
    predicted_correspondences.add((data["Élément 1"][i], data["Élément 2"][j]))

# Calculate true positives, false positives, and false negatives
true_positives = len(predicted_correspondences & true_correspondences)
false_positives = len(predicted_correspondences - true_correspondences)
false_negatives = len(true_correspondences - predicted_correspondences)

# Calculate precision, recall, and F1-score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print the distance table
print("SimHash Distance Table (5x5):\n")
header = [""] + data["Élément 2"]
print("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}".format(*header))
for i, row in enumerate(distance_table):
    print("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}".format(data["Élément 1"][i], *row))

print("\nMetrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")