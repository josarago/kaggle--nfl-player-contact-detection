
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import lightgbm as lgb


from config import (
	TRAIN_LABELS_PATH,
	TRAIN_TRACKING_PATH,
	SUBMISSION_REF_PATH,
	TEST_TRACKING_PATH,
	SUBMISSION_FILE_NAME
)

from utils import add_contact_id, expand_contact_id
from pipelines import tracking_pipeline

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
		self.model = None
		
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
		train_tracking_df = pd.read_csv(
			self.train_tracking_path,
			parse_dates=["datetime"],
			dtype={"nfl_player_id": "str"}
		)
		return train_labels_df, train_tracking_df
	
	def load_test_data(self):
		submission_ref_df = pd.read_csv(self.submission_ref_path)
		test_tracking_df= pd.read_csv(
			self.test_tracking_path,
			parse_dates=["datetime"],
			dtype={"nfl_player_id": "str"}
		)
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

		return feature_df

	# def get_training_set(self, df):
	# 	df_features = df[self._feature_columns]
	# 	y = df[self._target_columns].values
	# 	return df_features, y

	def init_model(self, params={}):
		if self._model_type == "decision_tree":
			print("creating Decision Tree classifier")
			self.model = DecisionTreeClassifier(**params)
		elif self._model_type == "lgbm":
			print("creating LightGBM classifier")
			self.model = lgb.LGBMClassifier(**params)


	def simple_train(self, X, y, params=None):
		self.init_model()
		self.model.fit(X, y)

	def grid_search(
			self,
			X,
			y,
			groups=None,
			param_grid=None
		):
		self.init_model()
		if groups is not None:
			cv = StratifiedGroupKFold(
				n_splits=CV_N_SPLITS,
				shuffle=False,
			)
		else:
			cv = StratifiedKFold(
				n_splits=CV_N_SPLITS,
			)
			
		_param_grid = param_grid if param_grid is not None else GRID_SEARCH_PARAM_GRID[self._model_type]
		self.clf = GridSearchCV(
			self.model,
			param_grid=_param_grid,
			scoring="roc_auc",
			cv=cv,
			n_jobs=-1,
			verbose=3
		)
		self.clf.fit(X, y, groups=groups)
		self._best_params = self.clf.best_params_

	def full_fit_with_best_params(self, X, y):
		self.init_model(params=self._best_params)
		self.model.fit(X, y)

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
		if write_file:
			print(f"Writing submission to: '{self._submission_file_name}'")
			submission_df.to_csv(self._submission_file_name, index=False)
		return submission_df