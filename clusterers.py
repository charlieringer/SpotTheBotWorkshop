import numpy as np
from data_loader import loadTimeSeries, loadFeatures

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#MAIN LOGIC
def main(args):
	#Load the data in what ever format was specified
	#Either ts - Time Series or features - using hand crafted features
	if args.data_mode == 'ts':
		x_train, y_train = loadTimeSeries(args.train_data)
		x_test, y_test = loadTimeSeries(args.test_data)

		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
	elif args.data_mode == 'features':
		x_train, y_train = loadFeatures(args.train_data)
		x_test, y_test = loadFeatures(args.test_data)
	else: 
		print("Error no data mode called ", args.mode, ". Exiting.")
		return

	if args.model == 'kmeans': model = KMeans(n_clusters=2)
	elif args.model == 'dbscan': model = DBSCAN()
	elif args.model == 'spec': model = SpectralClustering(n_clusters=2)
	else:
		print("Error no model called ", args.model, ". Exiting.")
		return
	preds = model.fit_predict(x_test)

	fig = plt.figure(0)
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	x_test_decomp = PCA(n_components=3).fit_transform(x_test)
	ax.scatter(x_test_decomp[:, 0], x_test_decomp[:, 1], x_test_decomp[:, 2],
	           c=preds.astype(np.float), edgecolor='k')

	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('PC 1')
	ax.set_ylabel('PC 2')
	ax.set_zlabel('PC 3')
	ax.set_title('Clusters')
	ax.dist = 12

	fig = plt.figure(1)
	ax2 = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	x_test_decomp = PCA(n_components=3).fit_transform(x_test)
	ax2.scatter(x_test_decomp[:, 0], x_test_decomp[:, 1], x_test_decomp[:, 2],
	           c=y_test.astype(np.float), edgecolor='k')

	ax2.w_xaxis.set_ticklabels([])
	ax2.w_yaxis.set_ticklabels([])
	ax2.w_zaxis.set_ticklabels([])
	ax2.set_xlabel('PC 1')
	ax2.set_ylabel('PC 2')
	ax2.set_zlabel('PC 3')
	ax.set_title('Ground Truth')
	ax2.dist = 12

	plt.show()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--test_data', dest='test_data')
	parser.add_argument('--train_data', dest='train_data')
	parser.add_argument('--data_mode', dest='data_mode', default='ts')
	parser.add_argument('--model', dest='model', default='knn')
	args = parser.parse_args()
	main(args)