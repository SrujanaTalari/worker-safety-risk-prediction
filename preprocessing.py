from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    encoder = LabelEncoder()
    for col in df.select_dtypes(include="object"):
        df[col] = encoder.fit_transform(df[col])
    X = df.drop("Safety_Risk_Level", axis=1)
    y = df["Safety_Risk_Level"]
    return train_test_split(X, y, test_size=0.25, random_state=42)