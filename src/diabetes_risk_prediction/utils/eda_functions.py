# Here we will store the functions for the EDA
# These functions will be used to inspect the data and the model
# The next functions are analytical functions:

import matplotlib.pyplot as plt
import seaborn as sns

def get_data_summary(df):
    """Prints a professional summary of the dataset's health."""
    print("--- DATASET SUMMARY ---")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().any().any() else "No missing values found.")
    print("\n--- DUPLICATES ---")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("-" * 60)
    
# The next functions are visual functions:

def plot_distribution(df, column):
    """Automatically chooses the best plot based on data type."""
    plt.figure(figsize=(6, 4))
    if df[column].dtype == 'object' or df[column].nunique() < 15:
        # Categorical or Discrete
        sns.countplot(data=df, x=column, palette='viridis', hue=column, legend=False)
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
    else:
        # Continuous Numerical
        sns.histplot(df[column], kde=True, color='skyblue')
        plt.title(f'Histogram & KDE of {column}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def check_outliers(df, column):
    """Uses Boxplots to identify extreme values in numerical features."""
    if df[column].dtype != 'object':
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[column], color='salmon')
        plt.title(f'Outlier Detection: {column}')
        plt.show()
    else:
        print(f"Skipping {column}: Outlier detection is for numerical data only.")
        
def plot_multiple_comparison_kde(df, features, target_col='diabetes_risk_category'):
    """
    Plots multiple KDE curves side-by-side in a single row for 
    easier feature comparison.
    """
    n_features = len(features)
    # Create a figure with 1 row and 'n' columns
    fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 5))
    
    # If only one feature is passed, axes is not a list, so we wrap it
    if n_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(features):
        sns.kdeplot(
            data=df, 
            x=feature, 
            hue=target_col, 
            fill=True, 
            common_norm=False, 
            palette='magma', 
            alpha=0.4,
            ax=axes[i]  # Plot on the specific subplot
        )
        axes[i].set_title(f'{feature} Distribution', fontsize=14)
        axes[i].set_xlabel(feature)
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()