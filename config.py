import os
from decouple import config as env_config

COMPETITION = "nfl-player-contact-detection"
ROOT_DIR = env_config('LOCAL_ROOT_DIR', default="../input")
BASE_DIR = os.path.join(ROOT_DIR, COMPETITION)

TRAIN_TRACKING_PATH = os.path.join(BASE_DIR, "train_player_tracking.csv")
TRAIN_LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")

TEST_TRACKING_PATH = os.path.join(BASE_DIR, "test_player_tracking.csv")
SUBMISSION_REF_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
SUBMISSION_FILE_NAME = "submission.csv"

MIN_X_POSITION = 0 
MAX_X_POSITION = 120

MIN_Y_POSITION = 0 
MAX_Y_POSITION = 53.3

if __name__ == "__main__":
    print(ROOT_DIR)