import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/sample_data.csv")

metadata_cols = ['N', 'Sample', 'Class', 'yClass']
spectral_cols = [c for c in df.columns if c not in metadata_cols]

cancer_classes = ['CBC', 'CEC', 'MEL']
df['binary_label'] = df['Class'].apply(lambda x: 1 if x in cancer_classes else 0)

X = df[spectral_cols]
y = df['binary_label']

# SNV
X = (X - X.mean(axis=1).values.reshape(-1,1)) / X.std(axis=1).values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
