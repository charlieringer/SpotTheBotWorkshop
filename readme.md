# Spot the Bot
## About
This repository supports the Spot the Bot Workshop at the 2018 IGGI Conference. It contains some data, in the form of game logs from humans and AIs playing video games as well as a selection of models for classifing these logs as human/AI as well as some clustering algorithms to explore the shape of the data.

## Data
Data has been gathered from a mixture of Humans and AI agents playing the GVGAI games Aliens, Frogs and SeaQuest. Half of the data set contains data from humans playing and the other half contains AI play. There is a folder (game_logs) which contains all of the raw data. There is also a csv, `data.csv`, which contains this all of this data in a format that the classifiers and clusterers want it in, generated using `log_parser.py`. Data was gathered using the in-built logging tool provided with GVGAI with a modification to the file name to include 'human' for human players and 'ai' for ai players as this is required for `log_parser.py` to be able to assign classification based on if a human or AI was playing. 

The `data.csv` file is in the following format:
- Rows: Each row is a different game log
- Columns:
    - 0 - Human/AI: 0 if AI, 1 if Human
    - 1 - Skill: 0 - Low Skill, 1 - Medium Skill, 2 - High Skill. This applies to humans (deterimened through self-reporting) and AI (based on GVGAI Rankings)
    - 2 - GameID: 0 - Aliens, 42 - Frogs, 77 - SeaQuest
    - 3 - LevelID: 0 - 4 depending on which level was played
    - 4 - PlayerID: Unique ID for each player/AI
    - 5 - Seed: Random seed assigned. Can be ignored
    - 6 - Won/Lost: 0 - Lost, 1 - Win
    - 7 - Game Score
    - 8 - Number of Game Ticks
    - 9 onwards - Moves for each tick where 0 - no move, 1 - left, 2 - right, 3 - up, 4 - down, 5 - use

There are two different ways this data can be loaded and passed to the algorithms, handled by  `data_loader.py`. Time-series data (`loadTimeSeries(file, testPercent)`) uses just the moves, in order, as input to the model. Because moves are categorical data (Left, Right etc.) they first must be 1-hot encoded (done by the data loader). Feature data (`loadFeatures(file, testPercent)`) uses a set of 9 hand crafted features which include Won/Lost, Game Score, Number of Game Ticks, % of 'nil' actions, % of 'left' actions,  % of 'right' actions, % of 'up' actions, % of 'down' actions, % of 'use' actions.

The `data_loader.py` methods also splits the data into training and test data based on a specified `testPercent`, all classification models provided use 80% training data and 20% test data.

Note for those devloping their own models: 1-hot encoding for time-series data is done in the "Keras" style, e.g. [[0,0,1],[1,0,0]..], and need to be converted to the "sk-learn" style e.g. [0,0,1,1,0,0..], for sk-learn models. Time series data also needs 0 padding for each row which is shorter than the longest row.

## Classification Algorithms
`classifiers.py` contains 7 different classification algorithms:
- k-Nearest Neighbour
- Logistic Regression
- Support Vector Classifier
- Decision Tree
- Gaussian Naive Bayes
- Multi-Layer Perceptron
- Recurrent Neural Network (GRU)

### Using classifiers.py
A typical command looks like this: `python3 classifiers.py --data=your_data.csv --data_mode=features --model=knn`. This will fit the model specified by `--model` on a test dataset built from 80% of the data in the file passed into `--data` using whichever data model is specifed by `--data_model`. It will then use this model to predict the class for the remaining 20% of the data (the test set) and based on this performance output some descriptive values based on the performance, including accuracy, percision, recall, specificity and a confusion matrix.

All models can be used on all data models with the exception of the Long Short-Term Memory Neural Network which only works with Time-Series data.

Possible arguments are:
- `--model`: `knn` - k-Nearest Neighbour, `logr` - Logistic Regression, `svc`- Support Vector Classifier, `dtree` - Decision Tree, `bayes` - Gaussian Naive Bayes, `mlp` - Multi-Layer Perceptron, `RNN` - Recurrent Neural Network.
- `--data_model`: `ts` - Time Series data, `features` - 9 hand crafted features (see Data for more details), `pca` - 3 features decomposed from the whole data set using PCA. NOT YET IMPLEMENTED
- `--data`: A .csv containing the player logs 

## Clusterering Algorithms
`clusterers.py` contains 3 different clustering algorithms:
- k-Means Clustering
- Birch Clustering
- Spectral Clustering

### Using clusterers.py
A typical command looks like this: `python3 clusterers.py --data=your_data.csv --data_mode=features --model=kmeans`. This will fit the model specified by `--model` on whole data set (ignoring classes) built from the data in the file passed into `--data` using whichever data model is specifed by `--data_model`. Then PCA is used to decompose the data into 3 features so it can be projected onto a 3d graph. The clustered data along with the ground truth data is then projected side by side into interactable 3d graphs, allowing you to explore the data.

All models can be used on all data models.

Possible arguments are:
- `--model`: `kmeans` - k-Means Clustering, `birch` - Birch Clustering, `spec`- Spectral Clustering
- `--data_model`: `ts` - Time Series data, `features` - 9 hand crafted features (see Data for more details), `pca` - 3 features decomposed from the whole data set using PCA. NOT YET IMPLEMENTED
- `--data`: A .csv containing the player logs 

## Dependencies
This workshop relies on the packages found in `requirements.txt`. This can be installed with the command `pip install -r requirements.txt`.

## Notes
More infomation about GVGAI can be found at: http://www.gvgai.net

All non-Neural Network models are implemented using scikit-learn - http://scikit-learn.org/

All Nueral Network models are implemented using Keras - https://keras.io/