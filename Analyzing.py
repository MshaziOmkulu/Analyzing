# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Optional: improve plot aesthetics
sns.set(style="whitegrid")

# ------------------------
# Task 1: Load and Explore the Dataset
# ------------------------

try:
    # Load Iris dataset from sklearn and convert to pandas DataFrame
    iris_raw = load_iris()
    iris = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    iris['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)
    
    # Show first 5 rows
    print("First 5 rows of the Iris dataset:")
    print(iris.head())
    
    # Check data types and missing values
    print("\nData types:")
    print(iris.dtypes)
    
    print("\nMissing values per column:")
    print(iris.isnull().sum())
    
    # No missing values in this dataset, but if there were, you could:
    # iris.fillna(method='ffill', inplace=True)  # or iris.dropna(inplace=True)

except Exception as e:
    print(f"Error loading data: {e}")

# ------------------------
# Task 2: Basic Data Analysis
# ------------------------

# Basic statistics for numerical columns
print("\nBasic statistics:")
print(iris.describe())

# Grouping by species and computing mean of numerical columns
print("\nMean measurements by species:")
print(iris.groupby('species').mean())

# Observations
print("\nObservation: Setosa generally has smaller petal length and width compared to Versicolor and Virginica.")

# ------------------------
# Task 3: Data Visualization
# ------------------------

# 1. Line chart: Sepal length trends by sample index (just an example since no time data)
plt.figure(figsize=(8,5))
for species in iris['species'].unique():
    subset = iris[iris['species'] == species]
    plt.plot(subset.index, subset['sepal length (cm)'], label=species)
plt.title('Sepal Length Trends by Sample Index')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar chart: Average petal length per species
plt.figure(figsize=(6,4))
sns.barplot(x='species', y='petal length (cm)', data=iris)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram: Distribution of sepal width
plt.figure(figsize=(6,4))
plt.hist(iris['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot: Sepal length vs Petal length colored by species
plt.figure(figsize=(7,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris)
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
