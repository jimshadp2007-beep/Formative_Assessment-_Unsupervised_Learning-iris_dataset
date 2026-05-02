# Formative_Assessment-_Unsupervised_Learning-iris_dataset

# 🌸 Iris Clustering Analysis

## 📌 Objective
The objective of this project is to evaluate and demonstrate the application of **clustering techniques** on a real-world dataset.

We apply:
- **KMeans Clustering**
- **Hierarchical Clustering**

to identify hidden patterns in the data without using labels.

---

## 📂 Dataset
This project uses the **Iris dataset** available in `sklearn`.

### Dataset Details:
- 150 samples
- 4 numerical features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

> ⚠️ Note: Since clustering is an **unsupervised learning task**, the species (target) column is not used.

---

## ⚙️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

## 🔹 1. Data Loading & Preprocessing

### Steps Performed:
- Loaded dataset using `sklearn.datasets`
- Converted dataset into a Pandas DataFrame
- Removed the target/species column

### Code:

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Use only feature data (drop target)
X = df.copy()

print(X.head())

#🔹 2. Clustering Algorithms

#🧠 A) KMeans Clustering
#📌 Description

KMeans is an unsupervised machine learning algorithm that:

#Divides the dataset into K clusters
#Assigns each data point to the nearest centroid
#Iteratively updates centroids to minimize within-cluster variance
#✅ Why KMeans for Iris?
#Works well with numerical data
#Iris dataset has naturally separable clusters
#Efficient and easy to implement
#⚙️ Implementation
#from sklearn.cluster import KMeans

# Apply KMeans
#kmeans = KMeans(n_clusters=3, random_state=42)
#kmeans_labels = kmeans.fit_predict(X)

# Add cluster labels to dataframe
#df['KMeans_Cluster'] = kmeans_labels
#📊 Visualization
#import seaborn as sns
#import matplotlib.pyplot as plt

#plt.figure(figsize=(8,5))

#sns.scatterplot(
#    x=df.iloc[:, 0],
#    y=df.iloc[:, 1],
 #   hue=df['KMeans_Cluster'],
 #   palette='viridis'
#)

#plt.title("KMeans Clustering")
#plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.show()
🌳 B) Hierarchical Clustering
📌 Description

Hierarchical clustering builds a hierarchy of clusters using:

Agglomerative approach (bottom-up) in this project
Starts with each data point as its own cluster
Merges closest clusters step-by-step
✅ Why Hierarchical for Iris?
No need to define number of clusters initially
Works well for small datasets
Provides visual representation using dendrogram
⚙️ Implementation
from sklearn.cluster import AgglomerativeClustering

# Apply Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3)
hc_labels = hc.fit_predict(X)

# Add cluster labels
df['HC_Cluster'] = hc_labels
📊 Visualization
plt.figure(figsize=(8,5))

sns.scatterplot(
    x=df.iloc[:, 0],
    y=df.iloc[:, 1],
    hue=df['HC_Cluster'],
    palette='coolwarm'
)

plt.title("Hierarchical Clustering")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.show()
🌿 Dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# Create linkage matrix
linked = linkage(X, method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked)

plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.show()
```
## 📊 Results & Insights
## Algorithm	Advantages	Limitations
## KMeans	Fast, simple, efficient	Requires predefined K
## Hierarchical Clustering	No need for initial K, interpretable	Slower for large datasets

# ✅ Conclusion
## Both clustering techniques successfully grouped the Iris dataset.
## KMeans is efficient when the number of clusters is known.
## Hierarchical Clustering provides better interpretability through dendrograms.
## The Iris dataset is well-suited for clustering due to its clear structure.

