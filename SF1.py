import hashlib

# Function to calculate the Hamming distance between two SimHashes
def hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count('1')

# Function to calculate the SimHash of a string
def simhash(value):
    hash_value = int(hashlib.md5(value.encode('utf-8')).hexdigest(), 16)
    return hash_value

# Provided data
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

# Calculate SimHashes for each element
simhashes_scrum = [simhash(elem) for elem in scrum_elements]
simhashes_prince2 = [simhash(elem) for elem in prince2_elements]

# Calculate Hamming distances and store in a 5x5 table
distance_table = [[hamming_distance(h1, h2) for h2 in simhashes_prince2] for h1 in simhashes_scrum]

# Define true correspondences
true_correspondences = {
    ("Product Backlog", "Product Description"),
    ("Sprint", "Stage Plan"),
    ("Daily Scrum", "Daily Standup"),
    ("Sprint Review", "End Stage Assessment"),
    ("Sprint Retrospective", "Lessons Learned")
}

# Determine predicted correspondences based on minimum distance
predicted_correspondences = set()
for i, row in enumerate(distance_table):
    min_distance = min(row)
    j = row.index(min_distance)
    predicted_correspondences.add((scrum_elements[i], prince2_elements[j]))

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
header = [""] + prince2_elements
print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format(*header))
for i, row in enumerate(distance_table):
    print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format(scrum_elements[i], *row))

print("\nMetrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")