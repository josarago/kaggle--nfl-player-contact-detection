# Kaggle competition: [1st and Future - Player Contact Detection](https://www.kaggle.com/competitions/nfl-player-contact-detection)

### Detect Player Contacts from Sensor and Video Data

## First approach: Using only sensor data
Each player can be, at any time, in contact with a variable number of player, the ground or nothing.

The simplest way to get started is to use only the sensor data, since they are tabular. Also it is apparently the solution currently used so it must be possible.

To predict a contact between players, we clearly need to use the sensor data for both of them: for instance a sustained contact probably implies a low relative speed. It is also likely that we want to predict the entire sequence of contact between two players at once, as the presence (or absence) of a contact at a given time is likely to affect the probability of a contact right after.


### Target variable
A player contact state can be represented as an array of size $N_players$

### Features
For each pair of players $(i, j)$ we want:
- the distance between the players
- the relative speed of the players
- the relative acceleration?

$x^i_t$:x position of player $i$ at time $t$
