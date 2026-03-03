# Here we will store the functions

import matplotlib.pyplot as plt
import seaborn as sns
import math

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
    
def plot_multiple_distributions(df_data, column_names):
    """
    Plots all specified columns in a single horizontal row.
    """
    num_plots = len(column_names)
    
    if num_plots == 0:
        print("No columns provided.")
        return

    # Create a figure with 1 row and 'n' columns
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    fig.suptitle('Side-by-Side Distribution Analysis', fontsize=16)

    # Ensure axes is always a list/array even if there is only 1 plot
    if num_plots == 1:
        axes = [axes]

    for i, col in enumerate(column_names):
        ax = axes[i]
        
        # Categorical vs Numerical Logic
        if df_data[col].dtype == 'object' or df_data[col].nunique() < 15:
            sns.countplot(data=df_data, x=col, palette='viridis', ax=ax, hue=col, legend=False)
            ax.set_title(f'Count: {col}')
            ax.tick_params(axis='x', rotation=45)
        else:
            sns.histplot(df_data[col], kde=True, color='skyblue', ax=ax)
            ax.set_title(f'KDE/Hist: {col}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_outliers_boxplot(df_data, column_names):
    """
    Genera diagramas de caja (box plots) para visualizar outliers en columnas numéricas.
    La función ajusta automáticamente el diseño de la cuadrícula.
    """
    num_plots = len(column_names)

    if num_plots == 0:
        print("No se proporcionaron columnas para graficar los outliers.")
        return

    # Determinar el número de filas y columnas para el subplots grid
    if num_plots == 1:
        nrows, ncols = 1, 1
    elif num_plots == 2:
        nrows, ncols = 1, 2
    elif num_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (num_plots + 1) // 3  # Intentar 3 columnas por fila para más de 4
        ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle('Outliers Detection with Box Plots', fontsize=16)

    # Aplanar los ejes para facilitar la iteración
    if num_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    for i, col in enumerate(column_names):
        if i < len(axes_flat):
            sns.boxplot(y=df_data[col], ax=axes_flat[i], color='lightsteelblue')
            axes_flat[i].set_title(f'Box Plot de {col}')
            axes_flat[i].set_ylabel(col)

    # Ocultar ejes vacíos si los hay
    for j in range(num_plots, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
        
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
    
    