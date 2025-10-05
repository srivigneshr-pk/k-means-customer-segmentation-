import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset (make sure the file is in the same folder)
df = pd.read_csv("archive/Mall_Customers.csv")

# Show the first 5 rows
print(df.head())

# Optional: Check column names
print(df.columns)

# Let's use Annual Income and Spending Score for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Create KMeans model with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Plot the clusters
plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], c=y_kmeans, cmap='rainbow')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.show()
