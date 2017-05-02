# Basic import
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.misc
from datetime import datetime

import tensorflow as tf

from git import Repo
if not os.path.exists('facenet'):
  Repo.clone_from("https://github.com/davidsandberg/facenet")

sys.path.insert(0, './facenet/src')
import align.detect_face
import facenet
import scipy.spatial.distance as distance

from IPython.display import Image, display

model_dir = '/home/sabrine/notebook/reid/20170216-091149/'

def detectionNetwork():
	# starting a session
	tf.reset_default_graph()

	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	tf_config.allow_soft_placement = True

	sess = tf.Session(config=tf_config)

	print('loading the detection/alignment network...')
	pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
	print('done!') 
	
	print('loading the embedding network...')
	meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
	restorer = tf.train.import_meta_graph(os.path.join(model_dir, meta_file))
	restorer.restore(sess, os.path.join(model_dir, ckpt_file))
	print('done!')

	# getting input / output tensors
	g = tf.get_default_graph()
	images_in = g.get_tensor_by_name('input:0')
	phase_train_in = g.get_tensor_by_name('phase_train:0')
	embeddings = g.get_tensor_by_name('embeddings:0')
	
	return sess, embeddings, images_in, phase_train_in, pnet, rnet, onet
