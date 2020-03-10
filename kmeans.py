import random
import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict
from itertools import combinations


def kmeans(X, k, max_iter=1000):
	centers=[tuple(c) for c in random.sample(X.tolist(), k)]#it randomly picks k elements from the input array X
	#X.tolist()-this is because X is a numpy array and so have to convert it into a list of values


	for i in range(max_iter):
		clusters=defaultdict(list)
		clusters_repid=defaultdict(list)
		repid=[]
		for value in X:
			ind=np.where(X==value)
			distance=[euclidean(value, x0) for x0 in centers]#find the euclidean distance between each value in X and each point in centers
			result=centers[np.argmin(distance)]#np.argmin returns the index of the minimum distance. find the value of the index in centers and append it to clusters cause that is the centroid
			clusters[result].append(value)
			clusters_repid[result].extend(list(ind))#list(ind) is because ind is also a numpy array returned by
			#statement np.where
			
			# get the index of value with re to the dataframe and use the index finally to pull the SalesRep_ID value
			# from the dataframe

		new_centers=[]
		for c, p in clusters.items():
			new_centers.append(tuple(np.mean(p, axis=0)))

		if set(new_centers)==set(centers):#if the earlier version of center the same as the newly created centers then break the loop
			break

		centers=new_centers#if not break then assign the newly created centers to the earlier version, i.e., centers and continue with the next iteration

	return clusters, clusters_repid


def sum_squared_error(clusters):
	sum_error=0
	for cen, point in clusters.items():
		for i in point:
			sum_error+=euclidean(i, cen)**2
	return sum_error


def turn_clusters_labels(clusters, clusters_repid, salesrep):
	l=0
	
	new_X=[]
	labels=[]
	for_repid=[]

	for center, point in clusters.items():
		for pt in point:
			new_X.append(pt)
	
			labels.append([l])#not sure whether [l] is necessary here

		l+=1

	for c, p in clusters_repid.items():
		for pts in p:
			for_repid.append(salesrep.loc[pts, 'SALESREP_ID'])

	return labels, new_X, for_repid#ideally it would be better to return lists so it is easier to use later in the
	#process especially to loop through and get salesrepid from a pandas dataset. it's not an easy task to do all this
	#using numpy arrays 













