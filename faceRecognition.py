from IPython.display import Image, display
import scipy.misc
import align.detect_face
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as distance
from munkres import Munkres

def embeddingKnownFaces(pathsKnown, names, sess, embeddings, 
	images_in, phase_train_in, pnet, rnet, onet, 
	minsize, threshold, factor):
 
	# Creation of an array of embedding known faces 
	# and an array of faces (image cropped to the center 
	# of the face detection) with the same index as the embedding array
	knownBoxes = []
	knownKeypoints = []
	knownFace = []
	print('detecting faces...')
	for path in pathsKnown:
		#display(Image(path))
		img = scipy.misc.imread(path)
		bbs, kps = align.detect_face.detect_face(img, minsize, 
												 pnet, rnet, onet, 
												 threshold, factor)
		kps = np.asarray(kps)
		print(kps.shape)
		kps = kps.reshape([2,5,-1]).T
		
		# Check if a face has been detected
		if len(kps) != 0:  
			# add the image in the face array
			knownFace.append(img)
			knownBoxes.append(bbs)
			knownKeypoints.append(kps)
			
			# Display the face only in order to check if the face was 
			# correctly detected
			margin = 32
			# Extract the position of the face from bbs
			for x0,y0,x1,y1,_ in bbs.astype(np.int32):
				x0 = np.maximum(x0 - margin//2, 0)
				y0 = np.maximum(y0 - margin//2, 0)
				x1 = np.minimum(x1 + margin//2, img.shape[1])
				y1 = np.minimum(y1 + margin//2, img.shape[0])
				#plt.imshow(img[y0:y1,x0:x1])
					 
	#print('done!')  
		
	# Computation of the embeddings for known faces
	#print('computing embeddings')
	size = 160
	margin = 32

	# Creation of the future array of embeddings for known faces and the
	# corresponding cropped faces for checking purpose
	knownFaces_embs = []
	knownCropped = []

	# Fill arrays
	for img, bbs in zip(knownFace, knownBoxes):
		img_faces = []
		# Cropped faces
		for x0,y0,x1,y1,_ in bbs.astype(np.int32):
			x0 = np.maximum(x0 - margin//2, 0)
			y0 = np.maximum(y0 - margin//2, 0)
			x1 = np.minimum(x1 + margin//2, img.shape[1])
			y1 = np.minimum(y1 + margin//2, img.shape[0])
		img_faces.append(scipy.misc.imresize(img[y0:y1,x0:x1], 
			(size, size)))
		img_faces = np.stack(img_faces)
		knownCropped.append(img_faces)
		#plt.imshow(img_faces[0])
	  
		# Embeddings
		feed_dict = {
			images_in : img_faces.astype(np.float32) / 255.0,
			phase_train_in : False,
		}
		knownFaces_embs.append(sess.run(embeddings, feed_dict))
	print('done!')
	
	if len(knownFaces_embs) != len(names):
		print("Error: the embedding faces and the name does not have the same lenght")
		return 0	
	else:
		data = np.vstack(knownFaces_embs)
		return pd.DataFrame(data,index=names)
		
def embeddingUnknownFaces(unknownPath, sess, embeddings, 
	images_in, phase_train_in, pnet, rnet, onet, 
	minsize, threshold, factor):
		
	# Creation of an array of faces that will be compared to known faces with an array of corresponding
	# faces (cropped images of the face detection)
	unknownFace = []
	#unknownBoxes = []
	unknownKeypoints = []
	
	img = scipy.misc.imread(unknownPath) 
	#display(Image(unknownPath))
	bbs, kps = align.detect_face.detect_face(img, minsize, 
												 pnet, rnet, onet, threshold, factor)
												
	kps = np.asarray(kps)
	#print(kps.shape)
	kps = kps.reshape([2,5,-1]).T
		
	# Check if a face has been detected
	if len(kps) != 0:  
		# add the image in the face array
		unknownFace.append(img)
		#unknownBoxes.append(bbs)
		unknownKeypoints.append(kps)
			
		# Display the face only in order to check if the face was corretly detected
		margin = 32
		# Extract the position of the face from bbs
		for x0,y0,x1,y1,_ in bbs.astype(np.int32):
			x0 = np.maximum(x0 - margin//2, 0)
			y0 = np.maximum(y0 - margin//2, 0)
			x1 = np.minimum(x1 + margin//2, img.shape[1])
			y1 = np.minimum(y1 + margin//2, img.shape[0])
			#plt.imshow(img[y0:y1,x0:x1])

	# Computation of the embeddings for unknown faces
	#print('computing embeddings')
	size = 160
	margin = 32

	# Creation of the future array of embeddings for unknown faces and 
	# the corresponding cropped faces for checking purpose
	unknownFaces_embs = []
	unknownCropped = []
	unknownFaceCenter =  []

	
	# Fill arrays
	for bb in bbs:
		img_faces = []
		# Cropped faces
		x0 = bb[0].astype(np.int32)
		y0 = bb[1].astype(np.int32)
		x1 = bb[2].astype(np.int32)
		y1 = bb[3].astype(np.int32)
		#for x0,y0,x1,y1,_ in bb: #.astype(np.int32):
		x0 = np.maximum(x0 - margin//2, 0)
		y0 = np.maximum(y0 - margin//2, 0)
		x1 = np.minimum(x1 + margin//2, img.shape[1])
		y1 = np.minimum(y1 + margin//2, img.shape[0])
		img_faces.append(scipy.misc.imresize(img[y0:y1,x0:x1], (size, size)))
		img_faces = np.stack(img_faces)
		unknownCropped.append(img_faces)
		unknownFaceCenter.append([(x1+x0)/2, (y1+y0)/2])
	  
		# Embeddings
		feed_dict = {
			images_in : img_faces.astype(np.float32) / 255.0,
			phase_train_in : False,
		}

		unknownFaces_embs.append(sess.run(embeddings, feed_dict))
	#print('done!')
	
	return unknownFaces_embs, unknownFaceCenter

def getNamesAndCoordinates(unknownFaces_embs, knownFaces_embs, unknownFaceCenter, names):
	# Creation of a dict 'name to index' in the unknownFaces_embs
	NtoI = dict()
	for e in set(names):
		NtoI[e] = [i for i, j in enumerate(names) if j == e]
		
	# return a matrix of euclidian distance pair by pairs
	dist = distance.cdist(unknownFaces_embs, knownFaces_embs)
                      
	'''  Construction of an array containing only the smallest value of all
    the distance between the embedding of the same person and 
    the embeddings of the detected faces 
    '''
	# Creation of an nan object as an empty array fo the smallest values 
	smallestArray = np.nan 
    
    # Loop over all the person of the database
	for key, value in NtoI.iteritems():
		
		# Creation of a vector of the smallest value for the different detection for one person
		test = np.vstack((dist[:, value[0]], dist[:, value[1]], dist[:, value[2]]))
		indexes = np.apply_along_axis(np.argmin, axis=1, arr=test.transpose())
    
		# Fill the array with the smallest value one person for each detection
		smallestSub = []
		for i, l in enumerate(test.T):
			smallestSub.append(l[indexes[i]])
		if np.isnan(smallestArray).all() == True:
			smallestArray = smallestSub
		else:
			smallestArray = np.vstack((smallestArray, smallestSub))
    	
	''' Usage of the Hungarian algorithm to minimze the sum of the 
	distance between matched embeddings'''
	
	# Creation of the Munkres object - can be used for several problem
	m = Munkres()
	
	# Feed the Munkres object with a copy 
	test = np.array(smallestArray.T)
	indexes = m.compute(test)
	
	n = []
	c = []
	d = []
	for row, column in indexes:
		center = unknownFaceCenter[row]
		c .append(center)
		#print('center : ', center)
		name = NtoI.keys()[column]
		n.append(name)
		#print('name : ', name)
		cost = smallestArray.T[row][column]
		d.append(cost)
		#print('cost : ', cost)
	
	#print(smallestArray.T)
	
	return n, c, d
	
def faceRecognition(pathsKnown, names, pathUnknown, sess, embeddings,
images_in, phase_train_in, pnet, rnet, onet, df):
	
	# detection parameters
	minsize = 20 
	threshold = [ 0.75, 0.75, 0.75 ]
	factor = 0.709
                   
	unknownFaces_embs, centerCoordinates = embeddingUnknownFaces(pathUnknown,
		sess, embeddings, images_in, phase_train_in, pnet, rnet, onet, 
		minsize, threshold, factor) 
		
	return getNamesAndCoordinates(np.vstack(unknownFaces_embs), pd.DataFrame.as_matrix(df), 
                      centerCoordinates, names)
	
	
