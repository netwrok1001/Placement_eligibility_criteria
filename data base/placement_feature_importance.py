"""Compute feature importances and SHAP-based influence, save CSV and TXT reports.
Outputs:
 - data base/features_importance.csv
 - data base/features_importance_summary.txt
"""
import joblib
import pandas as pd
import numpy as np
import shap

# paths
model_path = r"data base/placement_pipeline_tuned.joblib"
df_path = r"data base/student_placement_prediction_dataset_2026.csv"
out_csv = r"data base/features_importance.csv"
out_txt = r"data base/features_importance_summary.txt"

# load pipeline and data
pipe = joblib.load(model_path)
preprocessor = pipe.named_steps['preprocessor']
clf = pipe.named_steps['clf']
df = pd.read_csv(df_path)
if 'placed' not in df.columns:
    df['placed'] = df['placement_status'].map({'Placed':1, 'Not Placed':0})

exclude = ['placed','salary_package_lpa','placement_status','student_id']
feature_cols = [c for c in df.columns if c not in exclude]
X = df[feature_cols]
y = df['placed']

# get preprocessed array and feature names
X_pre = preprocessor.transform(X)
try:
    feat_names = preprocessor.get_feature_names_out(feature_cols)
except Exception:
    # fallback naming
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
    # try to expand onehot names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_names = list(ohe.get_feature_names_out(cat_cols)) if hasattr(ohe, 'get_feature_names_out') else cat_cols
    feat_names = list(num_cols) + list(ohe_names)

# shap explanation
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_pre)
# for binary classifier shap_values is list [neg, pos] or array; handle both
if isinstance(shap_values, list):
    shap_pos = shap_values[1]
else:
    shap_pos = shap_values

# compute importances
mean_abs_shap = np.abs(shap_pos).mean(axis=0)
mean_shap = shap_pos.mean(axis=0)
# model feature_importances_
try:
    model_imp = clf.feature_importances_
except Exception:
    model_imp = np.zeros_like(mean_abs_shap)

# assemble dataframe
df_imp = pd.DataFrame({
    'feature': feat_names,
    'mean_abs_shap': mean_abs_shap,
    'mean_shap': mean_shap,
    'lgb_importance': model_imp
})
# sort by mean_abs_shap desc
df_imp = df_imp.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

# direction: mean_shap >0 increases probability of Placed, <0 decreases
df_imp['influence'] = df_imp['mean_shap'].apply(lambda x: 'increasing' if x>0 else ('decreasing' if x<0 else 'neutral'))

# save CSV
df_imp.to_csv(out_csv, index=False)

# write summary TXT
with open(out_txt, 'w', encoding='utf-8') as f:
    f.write('Feature importance summary\n')
    f.write('==========================\n\n')
    f.write(f'Total transformed features: {len(df_imp)}\n')
    f.write('\nTop 20 features by mean(|SHAP|):\n')
    for i, row in df_imp.head(20).iterrows():
        f.write(f"{i+1}. {row['feature']}: mean_abs_shap={row['mean_abs_shap']:.6f}, mean_shap={row['mean_shap']:.6f}, lgb_imp={row['lgb_importance']} -> {row['influence']}\n")
    f.write('\nFeatures increasing model prediction for Placed (mean_shap>0):\n')
    inc = df_imp[df_imp['influence']=='increasing']
    for row in inc.itertuples():
        f.write(f"- {row.feature}\n")
    f.write('\nFeatures decreasing model prediction for Placed (mean_shap<0):\n')
    dec = df_imp[df_imp['influence']=='decreasing']
    for row in dec.itertuples():
        f.write(f"- {row.feature}\n")

print(f'Saved CSV: {out_csv}')
print(f'Saved TXT summary: {out_txt}')
