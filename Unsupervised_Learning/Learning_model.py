from Clustering import (
    compute_silhouette,
    create_clusters,
    get_cluster_names,
    print_cluster_examples,
    print_cluster_keywords,
)
from Dataset import load_dataset
from Preprocessing import vectorize

df = load_dataset("Student_Dataset.csv")

X, vectorizer = vectorize(df["text"])

model, labels = create_clusters(X, 7)

cluster_names = get_cluster_names(model, vectorizer)

print_cluster_keywords(model, vectorizer)

print_cluster_examples(df, labels, cluster_names)