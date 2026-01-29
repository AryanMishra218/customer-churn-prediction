import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print(" CUSTOMER CHURN PREDICTION - MODEL TRAINING")
print("="*70)

# STEP 1: LOAD DATA
print("\n Step 1: Loading dataset...")
df = pd.read_csv('data/bank_churn.csv')
print(f"✅ Loaded {len(df)} customer records")

# STEP 2: PREPARE DATA FOR ML
print("\n🔧 Step 2: Preparing data for machine learning...")

# Remove columns we don't need for prediction
# Why? CustomerId and Surname are just identifiers, not useful for prediction
df_model = df.drop(['CustomerId', 'Surname'], axis=1)
print(f"✅ Removed identifier columns")

# Convert categorical features to numbers
# Why? ML models only understand numbers, not text like "France" or "Male"
label_encoders = {}

# Encode Geography (France=0, Germany=1, Spain=2)
label_encoders['Geography'] = LabelEncoder()
df_model['Geography'] = label_encoders['Geography'].fit_transform(df_model['Geography'])

# Encode Gender (Female=0, Male=1)
label_encoders['Gender'] = LabelEncoder()
df_model['Gender'] = label_encoders['Gender'].fit_transform(df_model['Gender'])

print(f"✅ Converted text columns to numbers")
print(f"   - Geography: {dict(enumerate(label_encoders['Geography'].classes_))}")
print(f"   - Gender: {dict(enumerate(label_encoders['Gender'].classes_))}")

# STEP 3: SPLIT FEATURES AND TARGET
print("\n Step 3: Separating features and target...")

# X = all features (what we use to predict)
# y = target (what we want to predict: Churn)
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

print(f"✅ Features (X): {X.shape[1]} columns")
print(f"✅ Target (y): Churn (0=Stayed, 1=Left)")
print(f"\nFeature columns: {list(X.columns)}")

# STEP 4: SPLIT INTO TRAINING AND TESTING SETS
print("\n Step 4: Splitting data into train and test sets...")

# Why? We train on 80% data, test on 20% unseen data to check if model really works
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" Training set: {len(X_train)} customers ({len(X_train)/len(df)*100:.1f}%)")
print(f" Testing set: {len(X_test)} customers ({len(X_test)/len(df)*100:.1f}%)")

# STEP 5: SCALE FEATURES
print("\n Step 5: Scaling features...")

# Why? Features like Balance (0-250k) and Age (18-92) have different scales
# Scaling makes them comparable so model trains better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Features scaled to same range")

# STEP 6: TRAIN MODEL 1 - LOGISTIC REGRESSION
print("\n" + "="*70)
print("🔷 MODEL 1: LOGISTIC REGRESSION")
print("="*70)

print(" Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
lr_pred = lr_model.predict(X_test_scaled)

# Calculate metrics
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"\n Logistic Regression Results:")
print(f"    Accuracy:  {lr_accuracy*100:.2f}%")
print(f"    Precision: {lr_precision*100:.2f}%")
print(f"    Recall:    {lr_recall*100:.2f}%")
print(f"    F1-Score:  {lr_f1*100:.2f}%")

# STEP 7: TRAIN MODEL 2 - RANDOM FOREST
print("\n" + "="*70)
print(" MODEL 2: RANDOM FOREST")
print("="*70)

print(" Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test_scaled)

# Calculate metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"\nRandom Forest Results:")
print(f"    Accuracy:  {rf_accuracy*100:.2f}%")
print(f"    Precision: {rf_precision*100:.2f}%")
print(f"    Recall:    {rf_recall*100:.2f}%")
print(f"    F1-Score:  {rf_f1*100:.2f}%")

# STEP 8: TRAIN MODEL 3 - XGBOOST
print("\n" + "="*70)
print(" MODEL 3: XGBOOST")
print("="*70)

print(" Training XGBoost...")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
xgb_pred = xgb_model.predict(X_test_scaled)

# Calculate metrics
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print(f"\n XGBoost Results:")
print(f"    Accuracy:  {xgb_accuracy*100:.2f}%")
print(f"    Precision: {xgb_precision*100:.2f}%")
print(f"    Recall:    {xgb_recall*100:.2f}%")
print(f"    F1-Score:  {xgb_f1*100:.2f}%")

# STEP 9: COMPARE ALL MODELS
print("\n" + "="*70)
print("📊 MODEL COMPARISON")
print("="*70)

results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [lr_accuracy, rf_accuracy, xgb_accuracy],
    'Precision': [lr_precision, rf_precision, xgb_precision],
    'Recall': [lr_recall, rf_recall, xgb_recall],
    'F1-Score': [lr_f1, rf_f1, xgb_f1]
})

print("\n" + results_df.to_string(index=False))

# Find best model based on F1-Score
best_model_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_f1 = results_df.loc[best_model_idx, 'F1-Score']

print(f"\n BEST MODEL: {best_model_name} (F1-Score: {best_f1*100:.2f}%)")

# STEP 10: SAVE THE BEST MODEL
print("\n Step 10: Saving models and preprocessing objects...")

# Save all models
joblib.dump(lr_model, 'models/logistic_regression.pkl')
joblib.dump(rf_model, 'models/random_forest.pkl')
joblib.dump(xgb_model, 'models/xgboost.pkl')

# Save preprocessing objects (we need these for predictions later)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

print(f" All models saved to 'models/' folder")

# STEP 11: CREATE VISUALIZATIONS
print("\n Step 11: Creating comparison visualizations...")

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

axes[0].bar(x - width, results_df.iloc[0, 1:], width, label='Logistic Regression')
axes[0].bar(x, results_df.iloc[1, 1:], width, label='Random Forest')
axes[0].bar(x + width, results_df.iloc[2, 1:], width, label='XGBoost')

axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].set_ylim([0, 1])

# Plot 2: Confusion Matrix for best model
if best_model_name == 'Logistic Regression':
    best_pred = lr_pred
elif best_model_name == 'Random Forest':
    best_pred = rf_pred
else:
    best_pred = xgb_pred

cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title(f'Confusion Matrix - {best_model_name}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(['Stayed', 'Churned'])
axes[1].set_yticklabels(['Stayed', 'Churned'])

plt.tight_layout()
plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Comparison chart saved as 'data/model_comparison.png'")

# STEP 12: EXPLAIN METRICS
print("\n" + "="*70)
print("📚 UNDERSTANDING THE METRICS")
print("="*70)
print("""
ACCURACY: Out of all predictions, how many were correct?
   → Higher is better (but can be misleading with imbalanced data)

PRECISION: Out of customers we predicted would churn, how many actually churned?
   → Higher = fewer false alarms

RECALL: Out of all customers who actually churned, how many did we catch?
   → Higher = we catch more churners (fewer missed opportunities)

F1-SCORE: Balance between Precision and Recall
   → This is often the best metric for imbalanced datasets like ours!

CONFUSION MATRIX:
   - True Negatives (Top-Left): Correctly predicted "Stayed"
   - False Positives (Top-Right): Predicted "Churned" but actually Stayed
   - False Negatives (Bottom-Left): Predicted "Stayed" but actually Churned
   - True Positives (Bottom-Right): Correctly predicted "Churned"
""")

print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n All models trained and saved successfully!")
print(f" Best performing model: {best_model_name}")
print(f"\n Saved files:")
print(f"   - models/logistic_regression.pkl")
print(f"   - models/random_forest.pkl")
print(f"   - models/xgboost.pkl")
print(f"   - models/scaler.pkl")
print(f"   - models/label_encoders.pkl")