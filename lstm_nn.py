import keras
import numpy as np
import csv
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from data_loader import loadTimeSeries


def lstm_class(input_size):
    lstm_input = Input(input_size)

    x = LSTM(units=32, activation='sigmoid', return_sequences=True)(lstm_input)
    x = LSTM(units=32, activation='sigmoid', return_sequences=False)(x)
    x = Dense(units=1, activation='sigmoid')(x)
    model = Model(lstm_input, x)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def main(args):
	x, y = loadTimeSeries(args.data)
	model = lstm_class((x.shape[1],x.shape[2]))
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