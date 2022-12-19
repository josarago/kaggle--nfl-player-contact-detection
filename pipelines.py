import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

_RAW_TRACKING_FEATURES = (
    'x_position',
    'y_position',
    'speed',
    'direction',
    'orientation',
    'acceleration',
    'sa'
)

RAW_TRACKING_FEATURES = [f"{col}_1" for col in _RAW_TRACKING_FEATURES] + [f"{col}_2" for col in _RAW_TRACKING_FEATURES]

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
    

players_relative_speed_pipe = Pipeline(
    steps=[
        ("get_relative_speed", FunctionTransformer(get_relative_speed)),        
        ("scale", StandardScaler(with_mean=False)),
        ("impute", SimpleImputer(strategy="constant", fill_value=-1))
    ]
)

def players_are_same_team(df):
    return (df["team_1"]==df["team_2"]).values.reshape(-1, 1)

same_team_pipe = Pipeline(
    steps=[
        ("are_same_team", FunctionTransformer(players_are_same_team))
    ]
)

def is_ground_contact(df):
    return (df["nfl_player_id_2"]=="G").values.reshape(-1, 1)

is_ground_contact_pipe = Pipeline(
    steps=[
        ("is_ground_contact", FunctionTransformer(is_ground_contact))
    ]
)

raw_tracking_pipe = Pipeline(
    steps=[
        ("scale", StandardScaler(with_mean=True)),
        ("impute", SimpleImputer(strategy="constant", fill_value=-1))
    ]
)

raw_features_pipe = ColumnTransformer([("raw_tracking_features", raw_tracking_pipe, RAW_TRACKING_FEATURES)])

tracking_pipeline = FeatureUnion(
    [   
        ("raw_features", raw_features_pipe),
        ("players_distance", players_distance_pipe),
        ("relative_speed", players_relative_speed_pipe),
        ("same_team", same_team_pipe),
        ("is_ground_contact", is_ground_contact_pipe)
    ]

)