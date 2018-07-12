# Spot the Bot
## About
This repository supports the Spot the Bot Workshop at the 2018 IGGI Conference. It contains some data, in the form of game logs from humans and AIs playing video games as well as a selection of models for classifing these logs as human or AI as well as some clustering algorithms to explore the shape of the data.

## Data
Data has been gathers from a mixture of Humans and AI agents playing the GVGAI game Aliens. Half of the data set contains data from humans playing and the other half contains AI play. There is a folder (game_logs) which contains all of the raw data. There is also a csv (`data.csv`) which contains this all of this data in a format that the classifiers and clusterers want it in, generated using `log_parser.py`. Data was gathered using the in-built logging tool provided with GVGAI with a modification to the file name to include 'human' for human players and 'ai' for ai players as this is required for `log_parser.py` to beable to assign classification based on if a human or AI was playing. 

The `data.csv` file is in the following format:
- Rows: Each row is a different game log
- Coloums: 0 - Human or AI, 1 - Seed (can be ignored), 2 - Won/Lost, 3 - Game Score, 4 - Number of Game Ticks, 5 onwards - Moves for each tick where 0 - no move, 1 - left, 2 - right, 3 - up, 4 - down, 5 - use

There are a varity of different ways this data can be loaded and passed to the algorithms, handled by  `data_loader.py`. Time-series data uses just the moves, in order, as input to the model. Because moves are categorical data (Left, Right etc.) they first must be 1-hot encoded (done by the data loader). Feature data used a set of 9 hand crafted features which include Won/Lost, Game Score, Number of Game Ticks, % of 'nil' actions, % of 'left' actions,  % of 'right' actions, % of 'up' actions, % of 'down' actions, % of 'use' actions.

Note for those devloping their own models: 1-hot encoding for time-series data is done in the "Keras" style, e.g. [[0,0,1],[1,0,0]..], and need to be converted to the "sk-learn" style e.g. [0,0,1,1,0,0..], for sk-learn models.

## Classification Algorithms
`classifiers.py` contains 7 different classification algorithms:
- k-Nearest Neighbour
- Logistic Regression
- Support Vector Classifier
- Decision Tree
- Gaussian Naive Bayes
- Dense, Fully-Connected Neural Network
- Long Short-Term Memory Neural Network

### Using classifiers.py
A typical command looks like this: `python3 classifiers.py --data=your_data.csv --data_mode=features --model=knn`. This will fit the model specified by `--model` on a test dataset built from 80% of the data in the file passed into `--data` using whichever data model is specifed by `--data_model`. It will then use this model to predict the class for the remaining 20% of the data (the test set) and based on this performance output some descriptive values based on the performance, including accuracy, percision, recall, specificity and a confusion matrix.

All models can be used on all data models with the excepting of the Long Short-Term Memory Neural Network which only works with Time-Series data.

Possible arguments are:
- `--model`: `knn` - k-Nearest Neighbour, `logr` - Logistic Regression, `svc`- Support Vector Classifier, `dtree` - Decision Tree, `bayes` - Gaussian Naive Bayes, `denseNN` - Dense, Fully-Connected Neural Network, `lstmNN` - Long Short-Term Memory Neural Network.
- `--data_model`: `ts` - Time Series data, `features` - 9 hand crafted features (see Data for more details), `pca` - 3 features decomposed from the whole data set using PCA. NOT YET IMPLEMENTED
- `--data`: A .csv containing the player logs 


## Clusterering Algorithms
`clusterers.py` contains 3 different clustering algorithms:
- k-Means 
- Birch
- Spectral Clustering

## Notes
More infomation about GVGAI can be found at: http://www.gvgai.net
All non-Neural Network models are implemented using scikit-learn - http://scikit-learn.org/
All Nueral Network models are implemented using Keras - https://keras.io/