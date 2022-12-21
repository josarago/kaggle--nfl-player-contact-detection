import os
from decouple import config as env_config

from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter


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

OUTPUT_DIR = './'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_logger(use_file:bool = False, name=None):  
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter('[%(asctime)s](%(name)s) | %(levelname)s: %(message)s'))
    logger.addHandler(handler1)
    if use_file:
        filename = OUTPUT_DIR +'train'
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
    return logger

if __name__ == "__main__":
    print(ROOT_DIR)