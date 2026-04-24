"""Train placement prediction pipeline and run 5-fold CV.
Saves trained pipeline to data base/placement_pipeline.joblib
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# load data
path = r"data base/student_placement_prediction_dataset_2026.csv"
df = pd.read_csv(path)

# target
if 'placed' not in df.columns:
    df['placed'] = df['placement_status'].map({'Placed':1, 'Not Placed':0})

# drop identifier
if 'student_id' in df.columns:
    df = df.drop(columns=['student_id'])

# features and target
target = 'placed'
exclude = [target, 'salary_package_lpa', 'placement_status']
feature_cols = [c for c in df.columns if c not in exclude]

# numeric / categorical
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in numeric_cols]

print(f"Using {len(feature_cols)} features: {len(numeric_cols)} numeric, {len(cat_cols)} categorical")

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

# model: prefer LightGBM if available
try:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(random_state=42, n_estimators=200)
    model_name = 'LightGBM'
except Exception:
    model = HistGradientBoostingClassifier(random_state=42)
    model_name = 'HistGB'

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clf', model)])

X = df[feature_cols]
y = df[target]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy','precision','recall','f1','roc_auc']

print(f"Running 5-fold stratified CV with {model_name}...")
cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

print('\nCV results:')
for metric in scoring:
    vals = cv_results[f'test_{metric}']
    print(f"{metric}: mean={vals.mean():.4f}, std={vals.std():.4f}")

# fit final pipeline on full data and save
print('\nFitting final pipeline on full dataset...')
pipeline.fit(X, y)
model_path = r"data base/placement_pipeline.joblib"
joblib.dump(pipeline, model_path)
print(f"Saved trained pipeline to: {model_path}")
