# Customer Segmentation Using Machine Learning

## 1. Project Overview
Customer Segmentation is a machine learning project that groups customers based on their annual income and spending score. It helps businesses understand different types of customers and plan suitable marketing strategies.

## 2. Objectives
- Group customers with similar characteristics.
- Identify high-income and high-spending customers.
- Understand customer spending patterns.
- Generate customer segment summaries.
- Save the segmentation results to a CSV file.

## 3. Dataset
The project uses the Mall Customers dataset, stored in `Mall_Customers.csv`.

The dataset contains:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## 4. Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- PyCharm

## 5. Algorithm
K-Means Clustering is used to group customers into five clusters. StandardScaler is used to scale annual income and spending score before clustering.

## 6. Project Workflow
1. Load the dataset.
2. Examine the customer data.
3. Select annual income and spending score.
4. Scale the selected features.
5. Apply K-Means clustering.
6. Display the customer groups in a scatter plot.
7. Generate segment summaries and business insights.
8. Save the results as `Customer_Segmentation_Results.csv`.

## 7. Expected Outcome
The project identifies five customer groups based on income and spending behaviour. The results can help businesses understand customer types and make better marketing decisions.

## 8. How to Run
1. Install the required Python libraries.
2. Place `Mall_Customers.csv` in the project folder.
3. Run `customer_segmentation.py`.
4. View the cluster summary and graph.
5. Check `Customer_Segmentation_Results.csv` for the saved results.
