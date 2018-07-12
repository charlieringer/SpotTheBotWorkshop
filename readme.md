# Spot the Bot
## About
This repository supports the Spot the Bot Workshop at the 2018 IGGI Conference. It contains some data, in the form of game logs from humans and AIs playing video games as well as a selection of models for classifing these logs as human or AI as well as some clustering algorithms to explore the shape of the data.

## Data

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
A typical command looks like this: `python3 classifiers.py --data=your_data.csv --data_mode=features --model=knn`. This will fit the model specified by `--model` on a test dataset built from 80% of the data in the file passed into `--data` using whichever data model is specifed by `--data_model`. It will then use this model to predict the class for the remaining 20% of the data and based on this performance output some descriptive values based in the performance, including accuracy, percision, recall, specificity and a confusion matrix.

All models can be used on all data models with the excepting of the Long Short-Term Memory Neural Network which only works with Time-Series data.

Possible arguments are:
- `--model`: `knn` - k-Nearest Neighbour, `logr` - Logistic Regression, `svc`- Support Vector Classifier, `dtree` - Decision Tree, `bayes` - Gaussian Naive Bayes, `denseNN` - Dense, Fully-Connected Neural Network, `lstmNN` - Long Short-Term Memory Neural Network.
- `--data_model`: `ts` - Time Series data, `features` - 6 hand crafted features (see Data for more details)
- `--data`: A .csv containing the player logs 


## Clusterering Algorithms
`clusterers.py` contains 3 different clustering algorithms:
- k-Means 
- Birch
- Spectral Clustering