import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
file_path = './dataset_for_assignment_2.csv'
dataset = pd.read_csv(file_path)

# Data Exploration
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.histplot(dataset['Age'], kde=True)
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
sns.countplot(data=dataset, x='Gender')
plt.title('Gender Distribution')

plt.subplot(2, 2, 3)
sns.countplot(data=dataset, x='Activity Level')
plt.title('Activity Level Distribution')

plt.subplot(2, 2, 4)
sns.countplot(data=dataset, x='Location')
plt.title('Location Distribution')

plt.tight_layout()
plt.show()

# Basic statistics of the dataset
print(dataset.describe())

# Regression Analysis
X = dataset[['Age', 'Distance Travelled (km)', 'Calories Burned']]
y = dataset['App Sessions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Classification Analysis
X = dataset[['Age', 'Distance Travelled (km)', 'Calories Burned']]
y = dataset['Activity Level'].apply(lambda x: 1 if x == 'Active' else 0)  # Binarizing the activity level

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Clustering
features = dataset[['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

dataset['Cluster'] = clusters

plt.figure(figsize=(10, 7))
sns.scatterplot(data=dataset, x='Distance Travelled (km)', y='App Sessions', hue='Cluster', palette='viridis')
plt.title('User Clusters Based on App Usage Patterns')
plt.show()
