import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory for plots if it doesn't exist
plots_dir = 'eda_plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load the dataset
file_path = "data base\\student_placement_prediction_dataset_2026.csv"
df = pd.read_csv(file_path)

# 1. Basic Information
print("--- Basic Information ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# 2. Data Cleaning - Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Exploratory Data Analysis (EDA) - Visualizations

# Set the style for seaborn
sns.set_theme(style="whitegrid")

# a. Distribution of CGPA
plt.figure(figsize=(10, 6))
sns.histplot(df['cgpa'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of CGPA')
plt.xlabel('CGPA')
plt.ylabel('Frequency')
plt.savefig(os.path.join(plots_dir, 'cgpa_distribution.png'))
plt.close()

# b. Salary Package Distribution (for placed students)
plt.figure(figsize=(10, 6))
sns.histplot(df[df['salary_package_lpa'] > 0]['salary_package_lpa'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Salary Package (LPA) for Placed Students')
plt.xlabel('Salary Package (LPA)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(plots_dir, 'salary_distribution.png'))
plt.close()

# c. Placement Status by Branch
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='branch', hue='placement_status', palette='viridis')
plt.title('Placement Status by Branch')
plt.xlabel('Branch')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Placement Status')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'placement_by_branch.png'))
plt.close()

# d. Correlation Heatmap (for numerical features)
plt.figure(figsize=(16, 12))
numerical_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
plt.close()

# e. CGPA vs Placement Status (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='placement_status', y='cgpa', palette='Set2')
plt.title('CGPA vs Placement Status')
plt.xlabel('Placement Status')
plt.ylabel('CGPA')
plt.savefig(os.path.join(plots_dir, 'cgpa_vs_placement.png'))
plt.close()

# f. Coding Skill Score vs Placement Status
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='placement_status', y='coding_skill_score', palette='pastel')
plt.title('Coding Skill Score vs Placement Status')
plt.xlabel('Placement Status')
plt.ylabel('Coding Skill Score')
plt.savefig(os.path.join(plots_dir, 'coding_vs_placement.png'))
plt.close()

# g. Average Salary by College Tier
plt.figure(figsize=(10, 6))
avg_salary_tier = df[df['salary_package_lpa'] > 0].groupby('college_tier')['salary_package_lpa'].mean().reset_index()
sns.barplot(data=avg_salary_tier, x='college_tier', y='salary_package_lpa', palette='muted')
plt.title('Average Salary Package by College Tier')
plt.xlabel('College Tier')
plt.ylabel('Average Salary (LPA)')
plt.savefig(os.path.join(plots_dir, 'salary_by_tier.png'))
plt.close()
plt.show()

print(f"\nEDA completed. Plots saved in the '{plots_dir}' directory.")
