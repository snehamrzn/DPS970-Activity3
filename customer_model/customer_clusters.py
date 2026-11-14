# DPS970 Activity 3, Q11 to Q13: group customers by how they shop (clustering).
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load the data. Leave out Name, because it is only a label (Q11).
data = pd.read_csv("customers.csv")
X = data[["AverageSpend", "AverageFrequency"]]
print("Customers:", len(data))

# 2. Make 4 clusters (groups) of similar customers.
model = KMeans(n_clusters=4, n_init=10, random_state=42)
model.fit(X)
print("Silhouette score (how well the groups separate):", round(silhouette_score(X, model.labels_), 3))

# 3. Give each cluster a name from its centre: high or low spend, high or low frequency.
def cluster_name(spend, freq):
    if spend > 60 and freq > 28:
        return "Loyal Big Spenders"
    if spend > 60:
        return "Occasional Big Spenders"
    if freq > 28:
        return "Frequent Budget Shoppers"
    return "Casual Shoppers"

names = {}
print("\nThe 4 clusters (Q12):")
for i, (spend, freq) in enumerate(model.cluster_centers_):
    names[i] = cluster_name(spend, freq)
    size = (model.labels_ == i).sum()
    print(f"  Cluster {i}: {names[i]}  (avg spend {spend:.1f}, avg frequency {freq:.1f}, {size} customers)")

# 4. Predict the cluster for 3 new customers (Q13 screenshot).
new_customers = pd.DataFrame({
    "AverageSpend":     [90, 25, 30],
    "AverageFrequency": [47, 45,  8],
})
print("\nPredictions for 3 new customers (Q13):")
for (_, row), c in zip(new_customers.iterrows(), model.predict(new_customers)):
    print(f"  Spend {row.AverageSpend}, frequency {row.AverageFrequency} -> Cluster {c}: {names[c]}")
