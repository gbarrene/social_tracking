from IPython.display import Image, display
import scipy.misc
import align.detect_face
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as distance

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
		display(Image(path))
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
			#correctly detected
			margin = 32
			# Extract the position of the face from bbs
			for x0,y0,x1,y1,_ in bbs.astype(np.int32):
				x0 = np.maximum(x0 - margin//2, 0)
				y0 = np.maximum(y0 - margin//2, 0)
				x1 = np.minimum(x1 + margin//2, img.shape[1])
				y1 = np.minimum(y1 + margin//2, img.shape[0])
				plt.imshow(img[y0:y1,x0:x1])
					 
	print('done!')  
		
	# Computation of the embeddings for known faces
	print('computing embeddings')
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
	display(Image(unknownPath))
	bbs, kps = align.detect_face.detect_face(img, minsize, 
												 pnet, rnet, onet, threshold, factor)
												
	kps = np.asarray(kps)
	print(kps.shape)
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
			plt.imshow(img[y0:y1,x0:x1])

	# Computation of the embeddings for unknown faces
	print('computing embeddings')
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
	print('done!')
	
	return unknownFaces_embs, unknownFaceCenter

def getNamesAndCoordinates(unknownFaces_embs, knownFaces_embs, unknownFaceCenter, names):
	# return a matrix of euclidian distance pair by pairs
	dist = distance.cdist(unknownFaces_embs,
                      knownFaces_embs)
	
	# We look for the index that minimize the euclidian distance 
	# for each row of the euclidian distance pair by pairs matrix
	minIndex = np.apply_along_axis(np.argmin, axis=1, arr=dist)
	
	dictionary = dict()
	print('centerCoordinate', len(unknownFaceCenter))
	for i, index in enumerate(minIndex):
		print(index)
		k = names[index]
		print('name', k)
		v = unknownFaceCenter[i]
		print('center' , v)
		d = dist[i][index]
		print("distance", d)
		#dictionary[names[index]]  = unknownFaceCenter[index]
		
	return dictionary
	
