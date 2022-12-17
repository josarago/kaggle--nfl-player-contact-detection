import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer

def get_players_distance(df):
    return ((df["x_position_2"] - df["x_position_1"]) ** 2 +  \
        (df["y_position_2"] - df["y_position_1"]) ** 2).apply(np.sqrt).values.reshape(-1, 1)

players_distance_pipe = Pipeline(
    steps=[
        ("get_players_distance", FunctionTransformer(get_players_distance)),
        ("scale", StandardScaler(with_mean=False)),
        ("impute", SimpleImputer(strategy="constant", fill_value=-1))
    ]
)

def get_relative_speed(df):
    # def get_relative_speed(df):
    vx_1 = (df["speed_1"] * np.cos(df["direction_1"] * np.pi / 180)).values.reshape(-1, 1)
    vy_1 = (df["speed_1"] * np.sin(df["direction_1"] * np.pi / 180)).values.reshape(-1, 1)

    vx_2 = (df["speed_2"] * np.cos(df["direction_2"] * np.pi / 180)).values.reshape(-1, 1)
    vy_2 = (df["speed_2"] * np.sin(df["direction_2"] * np.pi / 180)).values.reshape(-1, 1)
    v = np.sqrt((vx_2 - vx_1)**2 + (vy_2 - vy_1)**2)
    return v

# def get_imputation_speed(v):
    

players_relative_speed_pipe = Pipeline(
    steps=[
        ("get_relative_speed", FunctionTransformer(get_relative_speed)),        
        ("scale", StandardScaler(with_mean=False)),
        ("impute", SimpleImputer(strategy="constant", fill_value=-1))
    ]
)

tracking_pipeline = FeatureUnion(
    [
        ("players_distance", players_distance_pipe),
        ("relative_speed", players_relative_speed_pipe)
    ]

)