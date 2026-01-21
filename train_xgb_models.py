import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

df = pd.read_csv("health_dataset_clinical_balanced_70_30.csv")

binary_targets = [
    'has_diabetes',
    'has_hypertension',
    'has_heart_disease',
    'has_kidney_disease',
    'has_liver_disease',
    'has_asthma'
]

# --- Feature Selection ---
X = df.drop(columns=binary_targets + ['patient_id'], errors='ignore')
X = X.select_dtypes(include=['int64', 'float64'])

# --- Impute Missing Values (VERY IMPORTANT) ---
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# --- Scale After Impute ---
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save transformers
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")

models = {}

for target in binary_targets:
    print("\nTraining:", target)

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- SMOTE Balancing ---
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # --- XGBoost Model ---
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, preds))

    models[target] = model

# Save Models
joblib.dump(models, "xgb_models.pkl")

print("\n All models saved successfully!")
