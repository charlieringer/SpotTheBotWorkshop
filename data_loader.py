import csv
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing import sequence

def getFeaturesForRow(data):
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

#Takes the raw data and returns a feature vector
#TODO: these first 3 features probably need scaling
def loadFeatures(file):
	loaded_file = open(file, 'r')
	data = csv.reader(loaded_file)
	rows = [[int(float(value)) for value in row if value] for row in data]
	x = []
	y = []
	for row in rows:
		y.append(row[0])
		x.append(getFeaturesForRow(row[2:]))

	x = np.array(x)
	y = np.array(y)
	
	return x, y

def loadNilActionPercent(file):
	loaded_file = open(file, 'r')
	data = csv.reader(loaded_file)
	rows = [[int(float(value)) for value in row if value] for row in data]
	x = []
	y = []
	for row in rows:
		y.append(row[0])
		features = []
		moves = row[3:]
		total_ticks = len(moves)
		features.append(moves.count(0)/total_ticks)
		x.append(features)

	x = np.array(x)
	y = np.array(y)
	
	return x, y

def loadTimeSeries(file):
	loaded_file = open(file, 'r')
	data = csv.reader(loaded_file)
	rows = [[int(float(value)) for value in row if value] for row in data]
	x = []
	y = []
	for row in rows:
		y.append(row[0])
		x.append(to_categorical(row[5:], num_classes=6))

	x = np.array(x)
	x = sequence.pad_sequences(x)
	y = np.array(y)
	
	return x, y