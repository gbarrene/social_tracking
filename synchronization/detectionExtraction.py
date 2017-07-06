''' Extraction of the timestamp, the seq number, the detection 
    coordinate of the 2D box the height and the distance from the Kinect
    of the detection '''

# Basic import
import rosbag
import numpy as np
import pandas as pd
from cv_bridge import CvBridge, CvBridgeError
import re

def detection_extraction(bagPath):
    
    # Create bag to process
    bag = rosbag.Bag(bagPath)
    
    # create bridge between CV and ROS
    br = CvBridge()
    
    # Creation of empty list thta will be fill by extracted information and then use to create a dataframe of useful info
    detectTime = []
    detectSeq = []
    
    for topic, msg, t in bag.read_messages(topics=['/detector/detections']):
        #print (str(msg).split('\n'))
        #break
        #for e in str(msg).split('\n'):
        #if 'Laurene' in e:

                
        # Extract timestamp
        time = msg.header.stamp
        detectTime.append(int(str(time)))
                
        #Extract seq number
        seq = msg.header.seq
        detectSeq.append(int(seq))
        
    #Creation of a dataframe of extracted information
    data = np.array([np.asarray(detectTime),
     np.asarray(detectSeq)]).T # The array need to be transposed
     
    detectionExtraction = pd.DataFrame(data=data)
    detectionExtraction.columns = ['detectionTimestamp', 'detecSeq']
            
    return detectionExtraction
    
