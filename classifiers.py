import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from data_loader import loadTimeSeries, loadFeatures
from keras.layers import Input, Dense, LSTM, GRU
from keras.models import Model

#MODEL
#For the Neural Nets we build out own model
def lstm_classifier(input_size):
    lstm_input = Input(input_size)

    x = GRU(units=32, activation='sigmoid', return_sequences=True)(lstm_input)
    x = GRU(units=32, activation='sigmoid', return_sequences=False)(x)
    x = Dense(units=1, activation='sigmoid')(x)
    model = Model(lstm_input, x)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#For the Neural Nets we build out own model
def dense_classifier(input_size):
    dense_input = Input(input_size)

    x = Dense(units=32, activation='sigmoid')(dense_input)
    x = Dense(units=32, activation='sigmoid')(x)
    x = Dense(units=1, activation='sigmoid')(x)
    model = Model(dense_input, x)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def knn_classifier():
	return KNeighborsClassifier()

def logr_classifier():
	return LogisticRegression()

def svc_classifier():
	return SVC()

def dtree_classifier():
	return DecisionTreeClassifier()

def bayes_classifier():
	return GaussianNB()

#MAIN LOGIC
def main(args):
	#Load the data in what ever format was specified
	#Either ts - Time Series or features - using hand crafted features
	if args.data_model == 'ts':
		x_train, y_train, x_test, y_test = loadTimeSeries(args.data, 0.8)
		if(args.model != 'lstm'):
			x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
			x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
	elif args.data_model == 'features':
		x_train, y_train, x_test, y_test = loadFeatures(args.data, 0.8)
	else: 
		print("Error no data mode called ", args.mode, ". Exiting.")
		return

	#Set the model to the right model depending on what arguement was passed
	if args.model == 'knn': model = knn_classifier()
	elif args.model == 'logr': model = logr_classifier()
	elif args.model == "svc": model = svc_classifier()
	elif args.model == "dtree": model = dtree_classifier()
	elif args.model == "bayes": model = bayes_classifier()
	elif args.model == 'mlp': model = dense_classifier((x_train.shape[1],))
	elif args.model == 'lstm': 
		if(args.data_model == 'features'):
			print("Error cannot use denseNN with feature data. Exiting.")
			return
		print(x_train.shape)
		model = lstm_classifier((x_train.shape[1],x_train.shape[2]))
	else: 
		print("Error no model called ", args.model, ". Exiting.")
		return

	#Here is where the magic happens, we fit the model to the training data
	if args.model == "lstm" or args.model == "mlp":
		model.fit(x_train,y_train,epochs=50, batch_size=10)
	else:
		model.fit(x_train,y_train)

	#Then predict the values for our test data
	preds = model.predict(x_test)

	#once we have our predictions we can evalute them
	#Start by building an empty confusion matrix
	confMat = [[0,0],[0,0]]

	#Them loop through our predictions and classifiy them (updating the conf matrix)
	for i in range(0, len(preds)):
		pred = 1 if (preds[i] > 0.5) else 0
		confMat[pred][y_test[i]] += 1

	#Work out the accuract
	totalCorrect = confMat[0][0] + confMat[1][1]
	total = len(preds)
	acc = totalCorrect/total
	prec = confMat[0][0]/ (confMat[0][0]+confMat[0][1])
	recall = confMat[0][0]/ (confMat[0][0]+confMat[1][0])
	spec = confMat[1][1]/ (confMat[0][1]+confMat[1][1])
	#And report this
	print("Testing Complete:")
	print("")
	print("Model Accuracy:    ", acc)
	print("Model Precision:   ", prec, " (w/resepct to AI)")
	print("Model Recall:      ", recall, " (w/resepct to AI)")
	print("Model Specificity: ", spec, " (w/resepct to AI)")
	print("")
	print("Confusion Matrix")
	print("   Actual AI   Hum")
	print("Pred AI  ",confMat[0][0],"  ", confMat[0][1])
	print("Pred Hum ",confMat[1][0],"  ", confMat[1][1])

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data', dest='data')
	parser.add_argument('--data_model', dest='data_model', default='ts')
	parser.add_argument('--model', dest='model', default='knn')
	args = parser.parse_args()
	main(args)