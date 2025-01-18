import json
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Étape 1 : Charger les fichiers JSON PRINCE2 et Scrum
prince2_data = {
    "PRINCE2": {
        "BusinessCase": {"ID": "String", "Title": "String", "Description": "String"},
        "ProjectBoard": {"ID": "String", "Name": "String", "Members": "List"},
        "ProjectPlan": {"ID": "String", "Name": "String", "StartDate": "Date", "EndDate": "Date"},
        "StagePlan": {"ID": "String", "Name": "String", "StageObjective": "String"},
        "WorkPackage": {"ID": "String", "Name": "String", "Tasks": "List"},
        "EndStageReport": {"ID": "String", "Name": "String", "Summary": "String"}
    }
}

scrum_data = {
    "Scrum": {
        "ProductBacklog": {"ID": "String", "Name": "String", "Description": "String"},
        "Sprint": {"ID": "String", "Name": "String", "Goal": "String", "StartDate": "Date", "EndDate": "Date"},
        "ScrumTeam": {"ID": "String", "Name": "String", "Members": "List"},
        "SprintBacklog": {"ID": "String", "Name": "String", "Tasks": "List"},
        "Increment": {"ID": "String", "Name": "String", "Description": "String", "Version": "String"},
        "DailyScrum": {"ID": "String", "Date": "Date", "Notes": "String"}
    }
}

# Étape 2 : Initialiser le modèle BERT multilingue et le tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Fonction pour obtenir les embeddings pour un terme ou une phrase
def get_word_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Fonction pour extraire et combiner les noms et descriptions des éléments du JSON
def extract_elements(data):
    combined_elements = []
    for key, attributes in data.items():
        combined_text = key + " " + " ".join([str(v) for v in attributes.values()])
        combined_elements.append(combined_text)
    return combined_elements

# Extraire les termes et descriptions pour PRINCE2 et Scrum
prince2_elements = extract_elements(prince2_data['PRINCE2'])
scrum_elements = extract_elements(scrum_data['Scrum'])

# Étape 3 : Générer les embeddings pour PRINCE2 et Scrum
prince2_embeddings = [get_word_embedding(text) for text in prince2_elements]
scrum_embeddings = [get_word_embedding(text) for text in scrum_elements]

# Étape 4 : Calculer la similarité cosinus entre les embeddings
cosine_sim = cosine_similarity(prince2_embeddings, scrum_embeddings)

# Étape 5 : Créer un DataFrame pour afficher la similarité
prince2_keys = list(prince2_data['PRINCE2'].keys())
scrum_keys = list(scrum_data['Scrum'].keys())
similarity_df = pd.DataFrame(cosine_sim, index=prince2_keys, columns=scrum_keys)

# Afficher le rapport de similarité
print("Rapport de similarité basé sur les Word Embeddings Multilingues :")
print(similarity_df)
from sklearn.metrics import precision_score, recall_score, f1_score

# Define a threshold for similarity (you can adjust this based on your needs)
threshold = 0.7

# Initialize variables for counting true positives, false positives, true negatives, and false negatives
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0

# Initialize a list to store predictions and true labels for metrics calculation
predictions = []
true_labels = []

# Loop through the cosine similarity matrix and calculate predictions
for i in range(len(prince2_keys)):
    for j in range(len(scrum_keys)):
        # If the cosine similarity is greater than or equal to the threshold, we predict a match (1)
        predicted = 1 if cosine_sim[i][j] >= threshold else 0
        
        # True correspondence: 1 if the elements are at the same index in both lists, otherwise 0
        true = 1 if i == j else 0
        
        # Update the counts for TP, FP, TN, FN
        if predicted == 1 and true == 1:
            true_positives += 1
        elif predicted == 1 and true == 0:
            false_positives += 1
        elif predicted == 0 and true == 1:
            false_negatives += 1
        elif predicted == 0 and true == 0:
            true_negatives += 1
        
        # Store predictions and true labels for later metric calculation
        predictions.append(predicted)
        true_labels.append(true)

# Calculate Precision, Recall, and F1-score
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# Print the metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Optionally, print the similarity table
print("\nSimilarity Table (Cosine Similarity):")
print(similarity_df)
