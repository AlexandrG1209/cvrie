from Clustering import (
    compute_silhouette,
    create_clusters,
    print_cluster_examples,
    print_cluster_keywords,
)
from Dataset import load_dataset
from Preprocessing import vectorize

df = load_dataset("Student_Dataset.csv")

X, vectorizer = vectorize(df["text"])

scores = compute_silhouette(X)

model, labels = create_clusters(X, 7)

print_cluster_keywords(model, vectorizer)

print_cluster_examples(df, labels)