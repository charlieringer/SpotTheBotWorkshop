import csv
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing import sequence

def __splitXonY(x,y):
	class0 = []
	class1 = []

	for i in range(0, len(x)):
		if(y[i] == 0): class0.append(x[i])
		else: class1.append(x[i])

	np.random.shuffle(class0)
	np.random.shuffle(class1)

	return class0, class1

def __getFeaturesForRow(data):
	features = []
	features.append(data[0])
	features.append(data[1])
	features.append(data[2])

	moves = data[3:]
	total_ticks = len(moves)
	features.append(moves.count(0)/total_ticks)
	features.append(moves.count(1)/total_ticks)
	features.append(moves.count(2)/total_ticks)
	features.append(moves.count(3)/total_ticks)
	features.append(moves.count(4)/total_ticks)
	features.append(moves.count(5)/total_ticks)

	return features

def __arrangeDataSets(x, y, trainPercent):
	class0, class1 = __splitXonY(x,y)

	splitat0 = int(len(class0) * trainPercent)
	splitat1 = int(len(class1) * trainPercent)

	trainX = class0[:splitat0] + class1[:splitat1]
	testX = class0[splitat0:] + class1[splitat1:]

	trainY = np.concatenate((np.zeros(splitat0, dtype=int), np.ones(splitat1, dtype=int)))
	testY = np.concatenate((np.zeros(len(class0)-splitat0, dtype=int), np.ones(len(class1)-splitat1, dtype=int)))

	trainX = np.array(trainX)
	trainY = np.array(trainY)

	testX = np.array(testX)
	testY = np.array(testY)

	permutation_train = np.random.permutation(trainX.shape[0])
	permutation_test = np.random.permutation(testX.shape[0])

	trainX = trainX[permutation_train]
	trainY = trainY[permutation_train]

	testX = testX[permutation_test]
	testY = testY[permutation_test]

	return trainX, trainY, testX, testY


#Takes the raw data and returns a feature vector
def load_features(file, trainPercent):
	loaded_file = open(file, 'r')
	data = csv.reader(loaded_file)
	next(data,None)
	rows = [[int(float(value)) for value in row if value] for row in data]
	x = []
	y = []
	for row in rows:
		y.append(row[0])
		x.append(__getFeaturesForRow(row[6:]))

	return __arrangeDataSets(x,y,trainPercent)

	
#Takes the raw data and returns a time series data
def load_time_series(file, trainPercent):
	loaded_file = open(file, 'r')
	data = csv.reader(loaded_file)
	next(data,None)
	rows = [[int(float(value)) for value in row if value] for row in data]
	x = []
	y = []
	for row in rows:
		y.append(row[0])
		x.append(to_categorical(row[9:], num_classes=6))

	x = np.array(x)
	x = sequence.pad_sequences(x)

	return __arrangeDataSets(x,y,trainPercent)