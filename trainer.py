
import pandas as pd

from config import (
	TRAIN_LABELS_PATH,
	TRAIN_TRACKING_PATH,
	SUBMISSION_REF_PATH,
	TEST_TRACKING_PATH
)

from utils import add_contact_id, expand_contact_id
from pipelines import tracking_pipeline



class ModelTrainer:
	def __init__(self,
		train_labels_path=TRAIN_LABELS_PATH,
		train_tracking_path=TRAIN_TRACKING_PATH,
		submission_ref_path=SUBMISSION_REF_PATH,
		test_tracking_path=TEST_TRACKING_PATH,
	):  
		self.train_labels_path = train_labels_path
		self.train_labels_df = None

		self.train_tracking_path = train_tracking_path
		self.train_tracking_df = None

		self.submission_ref_path = submission_ref_path
		self.submission_ref_df = None

		self.test_tracking_path = test_tracking_path		
		self.test_tracking_df = None

		self._tracking_pipeline = tracking_pipeline
	
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

	@staticmethod
	def join_tracking_data(
		contact_df: pd.DataFrame,
		tracking_df: pd.DataFrame,
		tracking_data_cols=["x_position", "y_position", "speed", "direction", "acceleration", "sa", "team", "position"],
		sample_frac=0.05
	):
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


	def train(self):
		pass

	def predict(self, X):
		pass
	
	def evaluate(self):
		pass

	@staticmethod
	def split_data(df_features, y, test_size, random_state=42):
		(
			df_features_train,
			df_features_test,
			y_train,
			y_test
		) = train_test_split(
			df_features,
			y,
			test_size=test_size,
			random_state=random_state
		)
		return df_features_train, df_features_test, y_train, y_test

	def make_submission_df(self, write_file=True):
		pass

if __name__ == "__main__":
	trainer = ModelTrainer()
	train_labels_df, train_tracking_df = trainer.load_training_data()

	train_feature_df = trainer.join_tracking_data(
		contact_df=train_labels_df,
		tracking_df=train_tracking_df,
		tracking_data_cols=["x_position", "y_position", "speed", "direction", "acceleration", "sa", "team", "position"],
 	   	sample_frac=None
	)

	X_train = trainer._tracking_pipeline.fit_transform(train_feature_df)
	print(X_train.shape)
	print(train_labels_df.shape)