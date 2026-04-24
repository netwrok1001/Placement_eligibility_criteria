"""Randomized hyperparameter search for LightGBM pipeline.
Saves best pipeline to data base/placement_pipeline_tuned.joblib
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
import joblib

# load data
path = r"data base/student_placement_prediction_dataset_2026.csv"
df = pd.read_csv(path)
if 'placed' not in df.columns:
    df['placed'] = df['placement_status'].map({'Placed':1, 'Not Placed':0})
if 'student_id' in df.columns:
    df = df.drop(columns=['student_id'])

# features
target = 'placed'
exclude = [target, 'salary_package_lpa', 'placement_status']
feature_cols = [c for c in df.columns if c not in exclude]

numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in numeric_cols]

# preprocessors
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', cat_transformer, cat_cols)
])

# LightGBM model
import lightgbm as lgb
base_model = lgb.LGBMClassifier(random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clf', base_model)])

X = df[feature_cols]
y = df[target]

# parameter distributions
param_dist = {
    'clf__n_estimators': [100, 200, 300, 500],
    'clf__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'clf__num_leaves': [31, 50, 80, 120],
    'clf__max_depth': [-1, 5, 10, 20],
    'clf__min_child_samples': [5, 10, 20, 40],
    'clf__subsample': [0.6, 0.8, 1.0],
    'clf__colsample_bytree': [0.6, 0.8, 1.0],
    'clf__reg_alpha': [0, 0.1, 0.5],
    'clf__reg_lambda': [0, 0.1, 0.5]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
rs = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=24,
    scoring='roc_auc',
    n_jobs=-1,
    cv=cv,
    random_state=42,
    verbose=2,
    refit=True
)

print('Starting RandomizedSearchCV (n_iter=24, cv=3)')
rs.fit(X, y)

print('\nBest score (roc_auc):', rs.best_score_)
print('Best params:')
for k, v in rs.best_params_.items():
    print(f" {k}: {v}")

# evaluate best estimator with 5-fold stratified CV
from sklearn.model_selection import cross_validate
scoring = ['accuracy','precision','recall','f1','roc_auc']
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(rs.best_estimator_, X, y, cv=cv5, scoring=scoring, n_jobs=-1)
print('\n5-fold CV (best estimator):')
for metric in scoring:
    vals = cv_results[f'test_{metric}']
    print(f"{metric}: mean={vals.mean():.4f}, std={vals.std():.4f}")

# save tuned pipeline
model_path = r"data base/placement_pipeline_tuned.joblib"
joblib.dump(rs.best_estimator_, model_path)
print(f"Saved tuned pipeline to: {model_path}")
