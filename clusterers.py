import numpy as np
from data_loader import loadTimeSeries, loadFeatures

from sklearn.cluster import KMeans, Birch, SpectralClustering
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#MAIN LOGIC
def main(args):
	#Load the data in what ever format was specified
	#Either ts - Time Series or features - using hand crafted features
	if args.data_model == 'ts':
		x_train, y_train, _, _ = loadTimeSeries(args.data, 1)
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
	elif args.data_model == 'features':
		x_train, y_train, _, _ = loadFeatures(args.data, 1)
	else: 
		print("Error no data mode called ", args.mode, ". Exiting.")
		return

	if args.model == 'kmeans': model = KMeans(n_clusters=2)
	elif args.model == 'birch': model = Birch(n_clusters=2)
	elif args.model == 'spec': model = SpectralClustering(n_clusters=2)
	else:
		print("Error no model called ", args.model, ". Exiting.")
		return

	preds = model.fit_predict(x_train)

	totalCorrect = 0
	for i, pred in enumerate(preds):
		if(y_train[i] == pred):
			totalCorrect+=1
	if(totalCorrect > len(y_train)-totalCorrect):
		print("Prediction score: ", totalCorrect/len(y_train))
	else:
		print("Prediction score: ", (len(y_train)-totalCorrect)/len(y_train))


	fig = plt.figure()
	ax = fig.add_subplot(1, 2, 1, projection='3d')

	x_test_decomp = PCA(n_components=3).fit_transform(x_train)
	ax.scatter(x_test_decomp[:, 0], x_test_decomp[:, 1], x_test_decomp[:, 2], c=preds.astype(np.float), edgecolor='k')

	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('PC 1')
	ax.set_ylabel('PC 2')
	ax.set_zlabel('PC 3')
	ax.set_title('Clusters')

	ax2 = fig.add_subplot(1, 2, 2, projection='3d')
	ax2.scatter(x_test_decomp[:, 0], x_test_decomp[:, 1], x_test_decomp[:, 2], c=y_train.astype(np.float), edgecolor='k')
	ax2.w_xaxis.set_ticklabels([])
	ax2.w_yaxis.set_ticklabels([])
	ax2.w_zaxis.set_ticklabels([])
	ax2.set_xlabel('PC 1')
	ax2.set_ylabel('PC 2')
	ax2.set_zlabel('PC 3')
	ax2.set_title('Ground Truth')

	
	plt.show()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data', dest='data')
	parser.add_argument('--data_model', dest='data_model', default='ts')
	parser.add_argument('--model', dest='model', default='kmeans')
	args = parser.parse_args()
	main(args)