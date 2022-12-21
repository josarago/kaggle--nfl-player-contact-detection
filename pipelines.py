import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def expand_feature_list(features):
    return [f"{col}_1" for col in features] + [f"{col}_2" for col in features]


_RAW_TRACKING_FEATURES = (
    'x_position',
    'y_position',
    'speed',
    'acceleration',
    'sa'
)
RAW_TRACKING_FEATURES = expand_feature_list(_RAW_TRACKING_FEATURES)

_ANGLE_FEATURES =  ('direction', 'orientation')
ANGLE_FEATURES = expand_feature_list(_ANGLE_FEATURES)



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

def make_cyclical_feature(x):
    _cos = np.cos(x * np.pi / 180)
    _sin = np.sin(x * np.pi / 180)
    return np.concatenate([_cos, _sin], axis=1)

cyclical_features_pipe = Pipeline(
    steps=[
        ("make_cyclical_features", FunctionTransformer(make_cyclical_feature)),
        ("impute", SimpleImputer(strategy="constant", fill_value=-2))
    ]
)

angle_features_pipe = ColumnTransformer([("raw_tracking_features", cyclical_features_pipe, ANGLE_FEATURES)])


def player_orientation_vs_direction(df):
    return pd.concat(
        [(df[f"orientation_{n}"] - df[f"direction_{n}"]) * np.pi / 180 for n in [1, 2]],
        axis=1
    ).apply(np.cos)

player_orientation_vs_direction_pipe = Pipeline(
    steps=[
        ("get_relative_speed", FunctionTransformer(player_orientation_vs_direction)),
        ("impute", SimpleImputer(strategy="constant", fill_value=-2))
    ]
)

def orientation_diff_cos(df):
    return (df["orientation_1"] - df["orientation_2"]).apply(np.cos).values.reshape(-1 ,1)

orientation_diff_cos_pipe = Pipeline(
    steps=[
        ("orientation_diff_cos", FunctionTransformer(orientation_diff_cos))
    ]
)


tracking_pipeline = FeatureUnion(
    [   
        ("raw_features", raw_features_pipe),
        ("angle_features", angle_features_pipe),
        ("players_distance", players_distance_pipe),
        ("relative_speed", players_relative_speed_pipe),
        ("same_team", same_team_pipe),
        ("is_ground_contact", is_ground_contact_pipe),
        ("player_orientation_vs_direction", player_orientation_vs_direction_pipe),
        ("orientation_diff_cos_pipe", orientation_diff_cos_pipe)
    ]

)