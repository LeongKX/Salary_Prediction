"""
Train a salary prediction model on the Latest Data Science Salaries dataset.

Key differences from the old notebook (why this one actually predicts):
  * No data leakage. We do NOT use Income_Category / Salary buckets as inputs,
    because those are derived from the salary itself.
  * Proper encoding. Categorical columns are One-Hot encoded instead of being
    turned into arbitrary integers with LabelEncoder.
  * The real drivers of salary are included: Experience Level, Expertise Level,
    Company Size, Job Title, etc.
  * The trained model AND everything the app needs (feature list, dropdown
    choices, metrics) are saved together into one .joblib file.

Run:  python train.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "Latest_Data_Science_Salaries.csv"
MODEL_PATH = "salary_model.joblib"
TARGET = "Salary in USD"

# Columns we let the model see. Note what is deliberately EXCLUDED:
#   - "Salary" and "Salary Currency": the raw salary in local currency -> leakage.
#   - "Salary in USD": that is the target.
CATEGORICAL = [
    "Job Title",
    "Employment Type",
    "Experience Level",
    "Expertise Level",
    "Company Location",
    "Company Size",
    "Employee Residence",
]
NUMERIC = ["Year"]
FEATURES = CATEGORICAL + NUMERIC


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",  # keeps Year as-is
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=24,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")

    print("=" * 55)
    print("Model performance (honest, no leakage)")
    print("=" * 55)
    print(f"Test R^2            : {r2:.3f}")
    print(f"Test MAE            : ${mae:,.0f}")
    print(f"5-fold CV R^2       : {cv_r2.mean():.3f} (+/- {cv_r2.std():.3f})")
    print(f"Median salary in USD: ${y.median():,.0f}")
    print("=" * 55)

    # Everything the Gradio app needs, saved in one file.
    choices = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL}
    choices["Year"] = sorted(int(v) for v in df["Year"].dropna().unique())

    artifact = {
        "model": model,
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "choices": choices,
        "metrics": {"r2": r2, "mae": mae, "cv_r2_mean": float(cv_r2.mean())},
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model + metadata -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
