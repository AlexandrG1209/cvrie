from Clustering import (
    compute_silhouette,
    create_clusters,
    print_cluster_examples,
    print_cluster_keywords,
)
from Dataset import load_dataset
from Preprocessing import vectorize, reduce_dimensions

df = load_dataset("Student_Dataset.csv")

X, vectorizer = vectorize(df["text"])

X_reduced, svd = reduce_dimensions(X)

scores = compute_silhouette(X_reduced)

print(
    f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}"
)
print(
    f"Silhouette score: {scores[7]:.4f}"
)

model, labels = create_clusters(X_reduced, 7)

print_cluster_keywords(model, vectorizer)

print_cluster_examples(df, labels)