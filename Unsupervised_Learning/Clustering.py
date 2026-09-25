from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_silhouette(X, min_clusters=2, max_clusters=20):

    scores = {}

    for k in range(min_clusters, max_clusters + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42
        )

        labels = model.fit_predict(X)

        scores[k] = silhouette_score(X, labels)

    return scores


def create_clusters(X, cluster_number):

    model = KMeans(
        n_clusters=cluster_number,
        random_state=42
    )

    labels = model.fit_predict(X)

    return model, labels

def get_cluster_names(model, vectorizer, word_number=2):
    feature_names = vectorizer.get_feature_names_out()
    cluster_names = {}

    for cluster in range(model.n_clusters):
        center = model.cluster_centers_[cluster]

        indices = center.argsort()[-word_number:][::-1]
        words = [feature_names[index] for index in indices]

        cluster_names[cluster] = " / ".join(words)

    return cluster_names


def print_cluster_keywords(model, vectorizer, word_number=10):

    feature_names = vectorizer.get_feature_names_out()

    for cluster in range(model.n_clusters):

        print("=" * 70)
        print(f"Cluster {cluster}")

        center = model.cluster_centers_[cluster]

        indices = center.argsort()[-word_number:][::-1]

        for index in indices:
            print(feature_names[index])


def print_cluster_examples(
    df,
    labels,
    cluster_names,
    example_number=5
):
    result = df.copy()
    result["cluster"] = labels

    for cluster in sorted(result["cluster"].unique()):

        print("=" * 70)
        print(
            f"Cluster {cluster}: "
            f"{cluster_names[cluster]}"
        )

        examples = result[
            result["cluster"] == cluster
        ]["text"]

        for example in examples.head(example_number):
            print("-", example)
