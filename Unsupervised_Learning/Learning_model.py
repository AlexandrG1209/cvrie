from pathlib import Path

import joblib

from Clustering import (
    compute_silhouette,
    create_clusters,
    get_cluster_names,
    print_cluster_examples,
    print_cluster_keywords,
)
from Dataset import load_dataset
from Preprocessing import vectorize

df = load_dataset("Unsupervised_Learning/Student_Dataset.csv")
Path("models/unsupervised").mkdir(parents=True, exist_ok=True)

X, vectorizer = vectorize(df["text"])

model, labels = create_clusters(X, 7)

cluster_names = get_cluster_names(model, vectorizer)

print_cluster_keywords(model, vectorizer)

print_cluster_examples(df, labels, cluster_names)

# Sauvegarder le modèle et les groupes pour le dashboard.
joblib.dump({"vectorizer": vectorizer, "model": model}, "models/unsupervised/text_model.joblib")
df.assign(cluster=labels).to_csv("models/unsupervised/patients.csv", index=False)
