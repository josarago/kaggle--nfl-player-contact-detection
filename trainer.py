import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer

import optuna

import lightgbm as lgb
import xgboost as xgb


from config import (
	TRAIN_LABELS_PATH,
	TRAIN_TRACKING_PATH,
	SUBMISSION_REF_PATH,
	TEST_TRACKING_PATH,
	SUBMISSION_FILE_NAME,
	get_logger
)

LOGGER = get_logger()

from utils import add_contact_id, expand_contact_id
from pipelines import tracking_pipeline


MATTHEWS_CORRCOEFF_SCORER = make_scorer(matthews_corrcoef)
	

TRACKING_DATA_COLS = (
	"x_position", 
	"y_position", 
	"speed", 
	"direction",
	"orientation",
	"acceleration", 
	"sa", 
	"team", 
	"position"
)

CV_N_SPLITS = 3

GRID_SEARCH_PARAM_GRID = dict()
GRID_SEARCH_PARAM_GRID["decision_tree"] = dict(
	criterion = ["gini", "entropy", "log_loss"],
	# splitter='best',
	max_depth=[1], 
	min_samples_split=[5, 10, 20],
	min_samples_leaf=[5, 10, 20],
	# min_weight_fraction_leaf=0.0,
	max_features=[1],
	# random_state=None,
	# max_leaf_nodes=None,
	# min_impurity_decrease=0.0,
	class_weight=["balanced"],
	# ccp_alpha=0.0
)

GRID_SEARCH_PARAM_GRID["lgbm"] = dict(
	num_leaves=[20, 30, 40],
	max_depth=[4, 5, 7],
	learning_rate=[0.005, 0.01, 0.02],
	n_estimators=[60, 80, 100],
	class_weight=["balanced"], 
	subsample=[0.8, 0.9], 
	# reg_alpha=[0.0, 0.01],
	# reg_lambda=[0.0, 0.01],
	n_jobs=[-1]
)


GRID_SEARCH_PARAM_GRID["xgb"] = dict(
	n_estimators=[20, 100, 150],
    learning_rate=[0.03, 0.05, 0.7]
)

PARAM_DISTRIBUTIONS = dict()
PARAM_DISTRIBUTIONS["xgb"] = dict(
    n_estimators=optuna.distributions.IntDistribution(10, 150, log=True),
    max_depth=optuna.distributions.IntDistribution(5, 20, log=False),
    max_leaves=optuna.distributions.IntDistribution(5, 100, log=True),
    learning_rate=optuna.distributions.FloatDistribution(0.005, 0.5, log=True),   
)

class ModelTrainer:
	def __init__(self,
		model_type,
		train_labels_path=TRAIN_LABELS_PATH,
		train_tracking_path=TRAIN_TRACKING_PATH,
		submission_ref_path=SUBMISSION_REF_PATH,
		test_tracking_path=TEST_TRACKING_PATH,
		submission_file_name=None
	):  
		self.train_labels_path = train_labels_path
		self.train_labels_df = None

		self.train_tracking_path = train_tracking_path
		self.train_tracking_df = None

		self._tracking_pipeline = tracking_pipeline
		self._model_type = model_type
		self.base_model = None
		self.model = None
		self.clf = None
		
		self.submission_ref_path = submission_ref_path
		self.submission_ref_df = None

		self.test_tracking_path = test_tracking_path		
		self.test_tracking_df = None

		self._submission_file_name = submission_file_name if submission_file_name else SUBMISSION_FILE_NAME

	def load_training_data(self):
		"""
			load all the data
		"""
		# training data
		train_labels_df = pd.read_csv(
			self.train_labels_path,
			parse_dates=["datetime"],
			dtype={"nfl_player_id_1": "str", "nfl_player_id_2": "str"}
		)
		LOGGER.info(f"`{self.train_labels_path}` file loaded")
		train_tracking_df = pd.read_csv(
			self.train_tracking_path,
			parse_dates=["datetime"],
			dtype={"nfl_player_id": "str"}
		)
		LOGGER.info(f"`{self.train_tracking_path}` file loaded")
		return train_labels_df, train_tracking_df
	
	def load_test_data(self):
		submission_ref_df = pd.read_csv(self.submission_ref_path)
		LOGGER.info(f"`{self.submission_ref_path}` file loaded")
		test_tracking_df= pd.read_csv(
			self.test_tracking_path,
			parse_dates=["datetime"],
			dtype={"nfl_player_id": "str"}
		)
		LOGGER.info(f"`{self.test_tracking_path}` file loaded")
		return submission_ref_df, test_tracking_df

	def join_tracking_data(
		self,
		contact_df: pd.DataFrame,
		tracking_df: pd.DataFrame,
		tracking_data_cols=list(TRACKING_DATA_COLS),
		sample_frac=None
	):
		self._tracking_data_cols = tracking_data_cols
		used_cols = tracking_data_cols + ["game_play", "step", "nfl_player_id"]

		if sample_frac is None or sample_frac == 1.0:
			_contact_df = contact_df
		else:
			LOGGER.info(f"Using subsanpling: sample_frac = {sample_frac} ")
			_contact_df = contact_df.sample(frac=sample_frac)

		feature_df = _contact_df \
			.merge(
				tracking_df[used_cols].rename(columns={col: f"{col}_1" for col in tracking_data_cols}),
				left_on=["game_play", "step", "nfl_player_id_1"],
				right_on=["game_play", "step", "nfl_player_id"],
			how="left") \
			.drop(columns=["nfl_player_id"]) \
			.merge(
				tracking_df[used_cols].rename(columns={col: f"{col}_2" for col in tracking_data_cols}),
				left_on=["game_play", "step", "nfl_player_id_2"],
				right_on=["game_play", "step", "nfl_player_id"],
			how="left") \
			.drop(columns=["nfl_player_id"])
		LOGGER.info("joined tracking data on label data")
		return feature_df

	def init_model(self, params={}):
		if self._model_type == "decision_tree":
			LOGGER.info("creating Decision Tree classifier")
			self.base_model = DecisionTreeClassifier(**params)
		elif self._model_type == "lgbm":
			LOGGER.info("creating LightGBM classifier")
			self.base_model = lgb.LGBMClassifier(**params)
		elif self._model_type == "xgb":
			LOGGER.info("creating XGBoost classifier")
			base_params = dict(
				tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
				objective="binary:logistic",
				eval_metric="auc"
			)
			self.base_model = xgb.XGBClassifier(
				**base_params,
				**params
			)

	def grid_search(
			self,
			X,
			y,
			groups=None,
			n_splits=CV_N_SPLITS,
			param_grid=None,
			scoring=MATTHEWS_CORRCOEFF_SCORER
		):
		self.init_model()
		if groups is not None:
			cv = StratifiedGroupKFold(
				n_splits=n_splits,
				shuffle=False,
			)
			LOGGER.info(f"Using `StratifiedGroupKFold` for cross validation")
		else:
			cv = StratifiedKFold(
				n_splits=n_splits,
			)
			LOGGER.info(f"Using `StratifiedKFold` for cross validation")
			
		_param_grid = param_grid if param_grid is not None else GRID_SEARCH_PARAM_GRID[self._model_type]
		self.clf = GridSearchCV(
			self.base_model,
			param_grid=_param_grid,
			scoring=scoring,
			cv=cv,
			n_jobs=-1,
			verbose=3,
			refit=True
		)
		LOGGER.info(f"Using `GridSearchCV` for hyper-parameters tuning: param_grid = {_param_grid}")
		self.clf.fit(X, y, groups=groups)
		LOGGER.info(f"Best parameters: {self.clf.best_params_}")
		self.model = self.clf.best_estimator_
		LOGGER.info(f"Model refit with best parameters")

	def optuna_search(self,
			X,
			y,
			groups=None,
			n_splits=CV_N_SPLITS,
			param_distributions=None,
			n_trials=100,
			scoring=MATTHEWS_CORRCOEFF_SCORER
		):
		LOGGER.info(f"training set shape: {X.shape}")
		self.init_model()
		if groups is not None:
			cv = StratifiedGroupKFold(
				n_splits=n_splits,
				shuffle=False,
			)
			LOGGER.info(f"Using `StratifiedGroupKFold` for cross validation with {n_splits} splits")
		else:
			cv = StratifiedKFold(
				n_splits=n_splits,
			)
			LOGGER.info(f"Using `StratifiedKFold` for cross validation with {n_splits} splits")
			
		_param_distributions = param_distributions if param_distributions is not None else PARAM_DISTRIBUTIONS[self._model_type]
		self.clf = optuna.integration.OptunaSearchCV(
			self.base_model,
			param_distributions=_param_distributions,
			n_trials=n_trials,
			scoring=scoring,
			cv=cv,
			n_jobs=-1,
			verbose=3,
			refit=True
		)
		LOGGER.info(f"Using optuna for hyper-parameters tuning: n_trials = {n_trials}, scoring: {scoring}")
		self.clf.fit(X, y, groups=groups)
		self.model = self.clf.best_estimator_
		LOGGER.info(f"Model refit with best parameters: {self.clf.best_params_}")

	def evaluate(self, y_true, y_pred):
		return matthews_corrcoef(y_true, y_pred)

	def make_submission_df(self, write_file=True):
		submission_ref_df, test_tracking_df = self.load_test_data()

		test_contact_df = expand_contact_id(submission_ref_df)
		test_feature_df = self.join_tracking_data(
			contact_df=test_contact_df,
			tracking_df=test_tracking_df,
			tracking_data_cols=self._tracking_data_cols,
			sample_frac=None
		)

		X_test = self._tracking_pipeline.transform(test_feature_df)
		submission_df = submission_ref_df.copy()
		submission_df["contact"] = self.model.predict(X_test)
		# submission
		submission_df = submission_df[['contact_id', 'contact']]
		LOGGER.info("Submission dataframe created")
		if write_file:
			LOGGER.info(f"Writing submission to: '{self._submission_file_name}'")
			submission_df.to_csv(self._submission_file_name, index=False)
		return submission_df