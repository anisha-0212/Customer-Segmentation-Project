
import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "Mall_Customers.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset size:")
print(df.shape)


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select features for customer segmentation
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCustomer segmentation completed!")
print("\nFirst 10 customers with their cluster:")
print(df[['CustomerID', 'Annual Income (k$)',
          'Spending Score (1-100)', 'Cluster']].head(10))

print("\nNumber of customers in each cluster:")
print(df['Cluster'].value_counts().sort_index())


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]

    plt.scatter(
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        label=f'Cluster {cluster}',
        s=70
    )

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Analyze each customer segment
segment_summary = df.groupby('Cluster').agg(
    Number_of_Customers=('CustomerID', 'count'),
    Average_Age=('Age', 'mean'),
    Average_Income=('Annual Income (k$)', 'mean'),
    Average_Spending_Score=('Spending Score (1-100)', 'mean')
).round(2)

print("\nCUSTOMER SEGMENT ANALYSIS")
print(segment_summary)


# Save customer segmentation results
output_path = "Customer_Segmentation_Results.csv"

df.to_csv(output_path, index=False)

print("\nCustomer segmentation results saved successfully!")
print("Saved file:", output_path)


# Give each segment a business description
summary = df.groupby('Cluster').agg(
    Customers=('CustomerID', 'count'),
    Average_Income=('Annual Income (k$)', 'mean'),
    Average_Spending=('Spending Score (1-100)', 'mean')
).round(2)

print("\nCUSTOMER SEGMENT INSIGHTS")

for cluster, row in summary.iterrows():
    income = row['Average_Income']
    spending = row['Average_Spending']

    if income >= 60 and spending >= 60:
        description = "High-income, high-spending customers"
    elif income >= 60 and spending < 40:
        description = "High-income, low-spending customers"
    elif income < 40 and spending >= 60:
        description = "Low-income, high-spending customers"
    elif income < 40 and spending < 40:
        description = "Low-income, low-spending customers"
    else:
        description = "Moderate-income or moderate-spending customers"

    print(f"\nCluster {cluster}: {description}")
    print(f"Number of customers: {int(row['Customers'])}")
    print(f"Average income: {income}")
    print(f"Average spending score: {spending}")
