import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")

# Load the dataset
print("📂 Loading dataset...")
df = pd.read_csv('data/bank_churn.csv')

print("\n" + "="*60)
print("📊 DATASET OVERVIEW")
print("="*60)

# Basic information
print(f"\n✅ Total Customers: {len(df)}")
print(f"✅ Total Features: {len(df.columns)}")
print(f"\n📋 Column Names:")
print(list(df.columns))

print("\n" + "="*60)
print("🔍 MISSING VALUES CHECK")
print("="*60)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values found!")

print("\n" + "="*60)
print("📈 CHURN STATISTICS")
print("="*60)
churn_counts = df['Churn'].value_counts()
churn_percentage = df['Churn'].value_counts(normalize=True) * 100

print(f"\n👥 Customers who STAYED (0): {churn_counts[0]} ({churn_percentage[0]:.2f}%)")
print(f"🚪 Customers who LEFT (1): {churn_counts[1]} ({churn_percentage[1]:.2f}%)")

print("\n" + "="*60)
print("📊 NUMERICAL FEATURES STATISTICS")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("🎯 CHURN BY DIFFERENT FACTORS")
print("="*60)

# Churn by Geography
print("\n🌍 Churn Rate by Country:")
churn_by_geo = df.groupby('Geography')['Churn'].mean() * 100
print(churn_by_geo.sort_values(ascending=False))

# Churn by Gender
print("\n👫 Churn Rate by Gender:")
churn_by_gender = df.groupby('Gender')['Churn'].mean() * 100
print(churn_by_gender.sort_values(ascending=False))

# Churn by Active Member
print("\n⚡ Churn Rate by Active Status:")
churn_by_active = df.groupby('Is Active Member')['Churn'].mean() * 100
print(churn_by_active.sort_values(ascending=False))

# Age analysis
print("\n🎂 Average Age:")
print(f"  - Customers who stayed: {df[df['Churn']==0]['Age'].mean():.1f} years")
print(f"  - Customers who left: {df[df['Churn']==1]['Age'].mean():.1f} years")

# Create visualizations
print("\n📊 Creating visualizations...")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Customer Churn Analysis', fontsize=16, fontweight='bold')

# 1. Churn Distribution
churn_counts.plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
axes[0, 0].set_title('Churn Distribution')
axes[0, 0].set_xlabel('Churn (0=Stayed, 1=Left)')
axes[0, 0].set_ylabel('Number of Customers')
axes[0, 0].set_xticklabels(['Stayed', 'Left'], rotation=0)

# 2. Churn by Geography
churn_by_geo.plot(kind='bar', ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Churn Rate by Country')
axes[0, 1].set_xlabel('Country')
axes[0, 1].set_ylabel('Churn Rate (%)')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

# 3. Churn by Gender
churn_by_gender.plot(kind='bar', ax=axes[0, 2], color='orange')
axes[0, 2].set_title('Churn Rate by Gender')
axes[0, 2].set_xlabel('Gender')
axes[0, 2].set_ylabel('Churn Rate (%)')
axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=0)

# 4. Age Distribution
df.boxplot(column='Age', by='Churn', ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution by Churn')
axes[1, 0].set_xlabel('Churn (0=Stayed, 1=Left)')
axes[1, 0].set_ylabel('Age')

# 5. Balance Distribution
df.boxplot(column='Balance', by='Churn', ax=axes[1, 1])
axes[1, 1].set_title('Balance Distribution by Churn')
axes[1, 1].set_xlabel('Churn (0=Stayed, 1=Left)')
axes[1, 1].set_ylabel('Balance ($)')

# 6. Credit Score Distribution
df.boxplot(column='CreditScore', by='Churn', ax=axes[1, 2])
axes[1, 2].set_title('Credit Score Distribution by Churn')
axes[1, 2].set_xlabel('Churn (0=Stayed, 1=Left)')
axes[1, 2].set_ylabel('Credit Score')

plt.tight_layout()
plt.savefig('data/churn_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'data/churn_analysis.png'")

print("\n" + "="*60)
print("🎯 KEY INSIGHTS")
print("="*60)
print("""
1. Class Imbalance: More customers stay than leave (this is normal)
2. Germany has highest churn rate among countries
3. Female customers churn more than male customers
4. Inactive members churn significantly more
5. Customers who leave are typically older
6. We need to handle these patterns when building our model!
""")

print("✅ Data exploration complete!")