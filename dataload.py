import pandas as pd
import numpy as np

def load_data():
    np.random.seed(42)
    data = {
        "Worker_Experience_Years": np.random.randint(1, 30, 300),
        "Weather_Condition": np.random.choice(
            ["Extreme Heat", "Foggy", "Rainy", "Clear"], 300
        ),
        "Construction_Task": np.random.choice(
            ["Road Work", "Metro Construction", "High-Rise Building", "Electrical"], 300
        ),
        "Safety_Training": np.random.choice(["Yes", "No"], 300),
        "Working_At_Height": np.random.choice(["Yes", "No"], 300),
        "Previous_Accidents": np.random.choice(["Yes", "No"], 300),
        "Safety_Risk_Level": np.random.choice(["Low", "Medium", "High"], 300),
    }
    return pd.DataFrame(data)