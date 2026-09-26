from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

root = Path(__file__).resolve().parent
results_dir = root / 'results'
results_dir.mkdir(exist_ok=True)

# ---- dataset detection ----
paths = [root / 'data' / 'Fraud.csv', root / 'data' / 'sample.csv']
path = next((p for p in paths if p.exists()), paths[0])
df = pd.read_csv(path)
print(f'Loaded dataset: {path.name}')

# ---- data cleanup ----
if 'nameOrig' in df.columns:
    df = df.drop(columns=['nameOrig', 'nameDest'], errors='ignore')
if 'isFlaggedFraud' in df.columns:
    df = df.drop(columns=['isFlaggedFraud'])

# feature engineering
for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
df['balanceChangeOrg'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['balanceChangeDest'] = df['newbalanceDest'] - df['oldbalanceDest']
df['amountToOldBalanceOrg'] = np.divide(
    df['amount'],
    df['oldbalanceOrg'].replace(0, np.nan),
    out=np.zeros(len(df), dtype=float),
    where=df['oldbalanceOrg'] != 0,
)
df['amountToOldBalanceDest'] = np.divide(
    df['amount'],
    df['oldbalanceDest'].replace(0, np.nan),
    out=np.zeros(len(df), dtype=float),
    where=df['oldbalanceDest'] != 0,
)
df['logAmount'] = np.log1p(df['amount'])

# dataset summary
summary = {
    'transactions': int(len(df)),
    'fraud_transactions': int(df['isFraud'].sum()),
    'legitimate_transactions': int((1 - df['isFraud']).sum()),
    'fraud_rate_percent': round(float(df['isFraud'].mean() * 100), 4),
    'features': int(df.drop(columns=['isFraud']).shape[1]),
    'missing_values': df.isna().sum().to_dict(),
    'duplicate_rows': int(df.duplicated().sum()),
    'transaction_types': df['type'].value_counts().to_dict(),
}
print(json.dumps(summary, indent=2))

# training features
X = df.drop(columns=['isFraud'])
y = df['isFraud'].astype(int)

# random split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

cat_cols = ['type'] if 'type' in X.columns else []
num_cols = [c for c in X.columns if c not in cat_cols]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ],
    remainder='drop',
)

# model definitions
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42, solver='liblinear')),
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)),
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=(y_train.value_counts().max() / y_train.value_counts().min()) if y_train.value_counts().min() else 1,
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}

# random-split evaluation
rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    rows.append({
        'Model': name,
        'Precision': float(precision_score(y_test, pred, zero_division=0)),
        'Recall': float(recall_score(y_test, pred, zero_division=0)),
        'F1': float(f1_score(y_test, pred, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(y_test, prob)),
        'PR-AUC': float(average_precision_score(y_test, prob)),
        'TP': int(((pred == 1) & (y_test == 1)).sum()),
        'TN': int(((pred == 0) & (y_test == 0)).sum()),
        'FP': int(((pred == 1) & (y_test == 0)).sum()),
        'FN': int(((pred == 0) & (y_test == 1)).sum()),
    })
model_df = pd.DataFrame(rows).sort_values('PR-AUC', ascending=False)
model_df.to_csv(results_dir / 'model_comparison.csv', index=False)
print(model_df.to_string(index=False))

best_name = model_df.iloc[0]['Model']
best_model = models[best_name]

# confusion matrix and classification report
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
print('Best model:', best_name)
print('Confusion matrix:')
print(cm)
print(classification_report(y_test, best_pred, target_names=['legitimate', 'fraud']))

# threshold analysis
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
prob_best = best_model.predict_proba(X_test)[:, 1]
threshold_rows = []
for t in thresholds:
    pred_t = (prob_best >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_t, labels=[0, 1]).ravel()
    threshold_rows.append({
        'threshold': t,
        'precision': float(precision_score(y_test, pred_t, zero_division=0)),
        'recall': float(recall_score(y_test, pred_t, zero_division=0)),
        'f1': float(f1_score(y_test, pred_t, zero_division=0)),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'true_negatives': int(tn),
    })
threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(results_dir / 'threshold_analysis.csv', index=False)

# plots
# class distribution
class_counts = df['isFraud'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Legitimate', 'Fraud'], [class_counts.get(0, 0), class_counts.get(1, 0)], color=['steelblue', 'darkorange'])
ax.set_title('Class Distribution')
ax.set_xlabel('Class')
ax.set_ylabel('Transactions')
fig.tight_layout(); fig.savefig(results_dir / 'class_distribution.png', dpi=200)
plt.close(fig)

# fraud by type
fraud_by_type = df.groupby('type')['isFraud'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
fraud_by_type.plot(kind='bar', color='seagreen', ax=ax)
ax.set_title('Fraud Rate by Transaction Type')
ax.set_xlabel('Transaction Type')
ax.set_ylabel('Fraud Rate')
fig.tight_layout(); fig.savefig(results_dir / 'fraud_by_type.png', dpi=200)
plt.close(fig)

# amount distribution
fig, ax = plt.subplots(figsize=(8, 5))
for label, group in df.groupby('isFraud'):
    sns.histplot(group['amount'], bins=25, alpha=0.7, label='Fraud' if label == 1 else 'Legitimate', ax=ax)
ax.set_title('Transaction Amount Distribution by Class')
ax.set_xlabel('Amount')
ax.set_ylabel('Count')
ax.legend()
fig.tight_layout(); fig.savefig(results_dir / 'amount_distribution.png', dpi=200)
plt.close(fig)

# roc curve
fig, ax = plt.subplots(figsize=(7, 6))
for name, pipe in models.items():
    probs = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
ax.set_title('ROC Curve Comparison')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
fig.tight_layout(); fig.savefig(results_dir / 'roc_curve.png', dpi=200)
plt.close(fig)

# precision-recall curve
fig, ax = plt.subplots(figsize=(7, 6))
for name, pipe in models.items():
    probs = pipe.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax.plot(recall, precision, label=f'{name} (AP={ap:.4f})')
ax.set_title('Precision-Recall Curve Comparison')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc='lower left')
fig.tight_layout(); fig.savefig(results_dir / 'precision_recall_curve.png', dpi=200)
plt.close(fig)

# confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'], ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title(f'Confusion Matrix - {best_name}')
fig.tight_layout(); fig.savefig(results_dir / 'confusion_matrix.png', dpi=200)
plt.close(fig)

# threshold plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(threshold_df['threshold'], threshold_df['precision'], marker='o', label='Precision')
ax.plot(threshold_df['threshold'], threshold_df['recall'], marker='s', label='Recall')
ax.plot(threshold_df['threshold'], threshold_df['f1'], marker='^', label='F1')
ax.set_title('Threshold Analysis')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Score')
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(results_dir / 'threshold_analysis.png', dpi=200)
plt.close(fig)

# feature importance
rf = models['Random Forest']
importances = rf.named_steps['model'].feature_importances_
feature_names = rf.named_steps['preprocessor'].get_feature_names_out()
fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1])
ax.set_title('Top 10 Features by Importance (Random Forest)')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
fig.tight_layout(); fig.savefig(results_dir / 'feature_importance.png', dpi=200)
plt.close(fig)

# time-based validation on the current sample dataset
if 'step' in df.columns:
    df_sorted = df.sort_values('step').reset_index(drop=True)
    median_step = df_sorted['step'].median()
    train_mask = df_sorted['step'] <= median_step
    test_mask = ~train_mask
    X_time_train = df_sorted.loc[train_mask, X.columns]
    y_time_train = df_sorted.loc[train_mask, 'isFraud'].astype(int)
    X_time_test = df_sorted.loc[test_mask, X.columns]
    y_time_test = df_sorted.loc[test_mask, 'isFraud'].astype(int)
    if len(np.unique(y_time_train)) > 1 and len(np.unique(y_time_test)) > 1 and len(X_time_train) > 0 and len(X_time_test) > 0:
        time_model = models['Logistic Regression']
        time_model.fit(X_time_train, y_time_train)
        prob_t = time_model.predict_proba(X_time_test)[:, 1]
        pred_t = time_model.predict(X_time_test)
        time_metrics = {
            'precision': float(precision_score(y_time_test, pred_t, zero_division=0)),
            'recall': float(recall_score(y_time_test, pred_t, zero_division=0)),
            'f1': float(f1_score(y_time_test, pred_t, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_time_test, prob_t)),
            'pr_auc': float(average_precision_score(y_time_test, prob_t)),
            'train_size': int(len(X_time_train)),
            'test_size': int(len(X_time_test)),
            'tp': int(((pred_t == 1) & (y_time_test == 1)).sum()),
            'tn': int(((pred_t == 0) & (y_time_test == 0)).sum()),
            'fp': int(((pred_t == 1) & (y_time_test == 0)).sum()),
            'fn': int(((pred_t == 0) & (y_time_test == 1)).sum()),
        }
    else:
        time_metrics = {
            'precision': None,
            'recall': None,
            'f1': None,
            'roc_auc': None,
            'pr_auc': None,
            'train_size': int(len(X_time_train)) if 'X_time_train' in locals() else 0,
            'test_size': int(len(X_time_test)) if 'X_time_test' in locals() else 0,
            'note': 'The sample dataset contains too few temporal steps for a stable time-based validation. The split is included to demonstrate the workflow, but the sample is not large enough to support a meaningful operational evaluation.',
        }
else:
    time_metrics = {'note': 'The current sample dataset does not contain a valid time column for temporal validation.'}

# metrics JSON
metrics = {
    'dataset': {
        'transactions': int(len(df)),
        'fraud_transactions': int(df['isFraud'].sum()),
        'legitimate_transactions': int((1 - df['isFraud']).sum()),
        'fraud_rate_percent': round(float(df['isFraud'].mean()*100), 4),
        'features': int(X.shape[1]),
        'source_file': path.name,
    },
    'models': {
        'logistic_regression': rows[0],
        'random_forest': rows[1],
        'xgboost': rows[2],
    },
    'selected_model': best_name,
    'threshold_analysis': threshold_df.to_dict(orient='records'),
    'time_based_validation': time_metrics,
}
(results_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

# notebook content
md_sections = [
    '# Fraud Detection Using Machine Learning',
    '## 1. Problem Statement\nThis project demonstrates a fraud-detection workflow for transaction-level financial data using classical machine learning and data analysis in a Jupyter notebook.',
    '## 2. Business Objective\nThe goal is to detect suspicious transactions while balancing precision and recall under severe class imbalance.',
    '## 3. Dataset Overview\nThis repository uses a portable local data path. The currently available dataset in the repository is a small sample, but the notebook is designed to work with a larger local file named `data/Fraud.csv` when available.',
    '## 4. Data Loading',
    '## 5. Data Quality Checks',
    '## 6. Exploratory Data Analysis',
    '## 7. Data Preprocessing',
    '## 8. Feature Engineering',
    '## 9. Class Imbalance Analysis',
    '## 10. Train-Test Split',
    '## 11. Model Training\n### 11.1 Logistic Regression\n### 11.2 Random Forest\n### 11.3 XGBoost',
    '## 12. Model Comparison',
    '## 13.Confusion Matrix',
    '## 14. ROC-AUC Analysis',
    '## 15. Precision-Recall Analysis',
    '## 16. Threshold Optimization',
    '## 17. Time-Based Validation',
    '## 18. Feature Importance',
    '## 19. Business Insights',
    '## 20. Limitations',
    '## 21. Conclusion',
]

nb = nbf.v4.new_notebook()
for section in md_sections:
    nb.cells.append(nbf.v4.new_markdown_cell(section))

code = '''
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

root = Path.cwd().resolve().parent
results_dir = root / 'results'
results_dir.mkdir(exist_ok=True)
path = next((p for p in [root / 'data' / 'Fraud.csv', root / 'data' / 'sample.csv'] if p.exists()), root / 'data' / 'sample.csv')
df = pd.read_csv(path)
print(f'Loaded dataset: {path.name}')
print(df.head())
print('shape', df.shape)
print('fraud_count', int(df['isFraud'].sum()))
print('legitimate_count', int((1 - df['isFraud']).sum()))
print('fraud_rate', df['isFraud'].mean())
print('missing', df.isna().sum().to_dict())
print('duplicates', int(df.duplicated().sum()))

if 'nameOrig' in df.columns:
    df = df.drop(columns=['nameOrig', 'nameDest'], errors='ignore')
if 'isFlaggedFraud' in df.columns:
    df = df.drop(columns=['isFlaggedFraud'])
for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
df['balanceChangeOrg'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['balanceChangeDest'] = df['newbalanceDest'] - df['oldbalanceDest']
df['amountToOldBalanceOrg'] = np.divide(df['amount'], df['oldbalanceOrg'].replace(0, np.nan), out=np.zeros(len(df), dtype=float), where=df['oldbalanceOrg'] != 0)
df['amountToOldBalanceDest'] = np.divide(df['amount'], df['oldbalanceDest'].replace(0, np.nan), out=np.zeros(len(df), dtype=float), where=df['oldbalanceDest'] != 0)
df['logAmount'] = np.log1p(df['amount'])

X = df.drop(columns=['isFraud'])
y = df['isFraud'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
cat_cols = ['type'] if 'type' in X.columns else []
num_cols = [c for c in X.columns if c not in cat_cols]
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
], remainder='drop')

models = {
    'Logistic Regression': Pipeline([('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42, solver='liblinear'))]),
    'Random Forest': Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))]),
    'XGBoost': Pipeline([('preprocessor', preprocessor), ('model', XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=(y_train.value_counts().max() / y_train.value_counts().min()) if y_train.value_counts().min() else 1, random_state=42, n_jobs=-1))]),
}
rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    rows.append({
        'Model': name,
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1': f1_score(y_test, pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, prob),
        'PR-AUC': average_precision_score(y_test, prob),
        'TP': int(((pred == 1) & (y_test == 1)).sum()),
        'TN': int(((pred == 0) & (y_test == 0)).sum()),
        'FP': int(((pred == 1) & (y_test == 0)).sum()),
        'FN': int(((pred == 0) & (y_test == 1)).sum()),
    })
results_df = pd.DataFrame(rows).sort_values('PR-AUC', ascending=False)
print(results_df)

best_name = results_df.iloc[0]['Model']
best_model = models[best_name]
print('Selected model:', best_name)
prob = best_model.predict_proba(X_test)[:, 1]
pred = best_model.predict(X_test)
print(confusion_matrix(y_test, pred, labels=[0, 1]))
print(classification_report(y_test, pred, target_names=['legitimate', 'fraud']))

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
threshold_rows = []
for t in thresholds:
    pred_t = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_t, labels=[0, 1]).ravel()
    threshold_rows.append({'threshold': t, 'precision': precision_score(y_test, pred_t, zero_division=0), 'recall': recall_score(y_test, pred_t, zero_division=0), 'f1': f1_score(y_test, pred_t, zero_division=0), 'false_positives': int(fp), 'false_negatives': int(fn)})
print(pd.DataFrame(threshold_rows))

# feature importance
rf = models['Random Forest']
fi_df = pd.DataFrame({'feature': rf.named_steps['preprocessor'].get_feature_names_out(), 'importance': rf.named_steps['model'].feature_importances_}).sort_values('importance', ascending=False).head(10)
print(fi_df)
print('Dataset summary', {'transactions': len(df), 'fraud_count': int(df['isFraud'].sum()), 'fraud_rate': df['isFraud'].mean()})
'''
nb.cells.append(nbf.v4.new_code_cell(code))
notebook_path = root / 'notebooks' / 'fraud_detection_analysis.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook saved at {notebook_path}')
