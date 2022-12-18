# Kaggle competition: [1st and Future - Player Contact Detection](https://www.kaggle.com/competitions/nfl-player-contact-detection)
### Detect Player Contacts from Sensor and Video Data
#
# First step: Using sensor data
Each player can be, at any time, in contact with a variable number of player, the ground or nothing.

The simplest way to get started is to use only the sensor data, since they are tabular. Also it is apparently the solution currently used so it must be possible.

To predict a contact between players, we clearly need to use the sensor data for both of them: for instance a sustained contact probably implies a low relative speed. It is also likely that we want to predict the entire sequence of contact between two players at once, as the presence (or absence) of a contact at a given time is likely to affect the probability of a contact right after.


## Submission 0
**Features**:
No feature

**Model**:
Dummy classifier: the contact is 1 with probability the mean value of the column `contact` in the `train_labels.csv` file

**Score**: 0.0

## Submission 1
**Feature**:
- distance between players

**Model**:
- Decision tree or SVM with gridsearch



#
# Cross validation strategy
- Contacts are likely to have some have amount of auto-correlation, meaning that if a player is in contact (resp. not in contact) with another player. If that is true, this suggests we should use K-fold cross validation with `game_play` as a group
- contacts are overall relatively rare: ~1.4% of all allowable contacts in the training set. If we use grouped K-fold cross-validation, we should probably consider insuring that contacts are not completely absent of some of the folds. If we train a binary classifier, it seems like a Stratified K-Folds with non-overlapping groups strategy is the most appropriate (`StratifiedGroupKFold`)




## Submission 1
**Features**:
- distance between players
- relative speed between players

**Model**: Decision Tree classifier, no hyperparameter tuning

**Score**: 0.373

## Submission 2
**Features**:
- distance between players
- relative speed between players
- are the players in the same team (boolean)
- is contact with ground (boolean)

**Model**: Decision Tree classifier, no hyperparameter tuning

**Score**: 0.442

## Submission 3
**Features**:
- distance between players
- relative speed between players
- are the players in the same team (boolean)
- is contact with ground (boolean)
- raw features: 
	`x_position`,
    `y_position`,
    `speed`,
    `direction`,
    `orientation`,
    `acceleration`,
    `sa`

**Model**: Decision Tree classifier, no hyperparameter tuning

**Score**: 0.462


