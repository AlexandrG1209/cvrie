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

def print_cluster_keywords(model, vectorizer, word_number=10):

    feature_names = vectorizer.get_feature_names_out()

    for cluster in range(model.n_clusters):

        print("=" * 70)
        print(f"Cluster {cluster}")

        center = model.cluster_centers_[cluster]

        indices = center.argsort()[-word_number:][::-1]

        for index in indices:
            print(feature_names[index])

def print_cluster_examples(df, labels, example_number=5):

    df = df.copy()

    df["cluster"] = labels

    for cluster in sorted(df["cluster"].unique()):

        print("=" * 70)
        print(f"Cluster {cluster}")

        examples = df[df["cluster"] == cluster]["text"]

        for example in examples.head(example_number):

            print("-", example)

"""
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def load_dataset(filename): # Load the patient dataset and return it as a pandas DataFrame.
    
    rows = []
    with open(filename, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        for row in reader:

            if len(row) == 2:
                rows.append({
                    "id": int(row[0]),
                    "color": None,
                    "text": row[1]
                })

            elif len(row) == 3:
                rows.append({
                    "id": int(row[0]),
                    "color": row[1],
                    "text": row[2]
                })

    return pd.DataFrame(rows)
"""
    

"""
df = load_dataset("Student_Dataset.csv")
texts = df["text"]
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_df=0.8,
    min_df=2,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts)

scores = []

for k in range(2, 8):

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    scores.append(score)

    print(f"{k} clusters -> {score:.4f}")

# ---------------------------------------------------

kmeans = KMeans(
    n_clusters=7,
    random_state=42
)

df["cluster"] = kmeans.fit_predict(X)

feature_names = vectorizer.get_feature_names_out()

for cluster in range(kmeans.n_clusters):

    print("=" * 60)
    print(f"Cluster {cluster}")

    center = kmeans.cluster_centers_[cluster]

    id = center.argsort()[-10:][::-1]

    for index in id:
        print(feature_names[index])


for cluster in sorted(df["cluster"].unique()):

    print("=" * 80)
    print(f"CLUSTER {cluster}")

    samples = df[df["cluster"] == cluster]["text"].head(5)

    for text in samples:
        print("-", text)



plt.plot(range(2, 8), scores, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title("Choosing the number of clusters")
plt.grid(True)

plt.show()
"""