import pandas as pd

def get_feature_importance(model, X):
    return pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)