import cv2
import numpy as np

cam = cv2.VideoCapture(0)


#img = cam.read()[1]
img = cam.read()[1]
pixel_used = []
x1=0
y1=0


while cam.isOpened():
	img = cam.read()[1]
	pixel_used.clear()
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	cv2.imshow("original",img)
	width = img.shape[0]
	height = img.shape[1]
	init_point = img[x1:(x1+1), y1:(y1+1)]
	for i in range(0,width):
		for j in range(0,height):
			next_point = img[i:(i+1),j:(j+1)]
			difference = next_point[0]-init_point[0]
			if(difference[0][0] <=10 and difference[0][1] <=10):
				pixel_used.append(next_point)

	'''
	while(difference[0][0] <=10 and difference[0][1] <=10 and difference[0][2] <=10):
		init_point = img[x1:(x1+1), y1:(y1+1)]
		next_point = img[x1:(x1+1),y1:(y1+1)]
		difference = next_point[0]-init_point[0]
		x1+=1
		#pixel_used[x1,y1] = True
	'''	
	img = img[0:x1+1,0:y1+1]
	print(len(pixel_used))	
	img = cv2.resize(img,(0,0),fx=100,fy=100)
	cv2.imshow("test?",img)
	k = cv2.waitKey(10)
	if k == 27:
   		break

'''

import numpy as np
from sklearn.cluster import KMeans
import argparse
import cv2
import datetime




image = cv2.imread("ed_wall.jpeg")
image = cv2.resize(image,(0,0),fx=.5,fy=.5)
orig = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
channels = cv2.split(image)
channelIndices = []
channelIndices.append(2)
image = image[:,:,channelIndices]
if len(image.shape) == 2:
    image.reshape(image.shape[0], image.shape[1], 1)
reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

numClusters = 5

kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped)

clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
    (image.shape[0], image.shape[1]))

sortedLabels = sorted([n for n in range(numClusters)],
    key=lambda x: -np.sum(clustering == x))
kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
for i, label in enumerate(sortedLabels):
    kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i

# Concatenate original image and K-means image, separated by a gray strip.
concatImage = np.concatenate((orig,
    193 * np.ones((orig.shape[0], int(0.0625 * orig.shape[1]), 3), dtype=np.uint8),
    cv2.cvtColor(kmeansImage, cv2.COLOR_GRAY2BGR)), axis=1)
while True:
	k = cv2.waitKey(10)
	if k == 27:
		break
	cv2.imshow('Original vs clustered', concatImage)
'''