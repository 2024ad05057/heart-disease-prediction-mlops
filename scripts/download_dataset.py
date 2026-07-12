import pandas as pd

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

df = pd.read_csv(
    "../data/raw/processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)

df.to_csv("../data/raw/heart.csv", index=False)

print("CSV created successfully!")