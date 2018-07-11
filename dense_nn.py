import csv
import numpy as np
from data_loader import loadFeatures
from keras.layers import Input, Dense
from keras.models import Model


def dense_class(input_size):
    dense_input = Input(input_size)

    x = Dense(units=32, activation='sigmoid')(dense_input)
    x = Dense(units=32, activation='sigmoid')(x)
    x = Dense(units=1, activation='sigmoid')(x)
    model = Model(dense_input, x)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def main(args):
	x, y = loadFeatures(args.data)
	model = dense_class((x.shape[1],))
	model.fit(x,y,epochs=50, batch_size=10)
	results = model.predict(x)

	confMat = [[0,0],[0,0]]

	for i in range(0, len(results)):
		pred = 1 if (results[i] > 0.5) else 0
		confMat[pred][y[i]] += 1

	totalCorrect = confMat[0][0] + confMat[1][1]
	total = len(results)
	acc = totalCorrect/total
	print("Model Accuracy: ", acc)
	print("          AI   Hum")
	print(" Pred AI ",confMat[0][0],"  ", confMat[0][1])
	print("Pred Hum ",confMat[1][0],"  ", confMat[1][1])

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data', dest='data')
	args = parser.parse_args()
	main(args)