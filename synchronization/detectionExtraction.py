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
    box2dList = []
    xList = []
    yList = []
    wList = []
    hList = []
    heightList = []
    distanceList = []
    
    for topic, msg, t in bag.read_messages(topics=['/detector/detections']):
        #print (str(msg).split('\n'))
        #break
        #for e in str(msg).split('\n'):
        #if 'Laurene' in e:

        # Extract the detection field that start and end with a regular pattern 
        begin = str(msg).find('detections:')
        end = str(msg).find('intrinsic_matrix:')
        detections = str(msg)[begin:end]

        # Extract the several detection for this msg
        detection = detections.split("- ")
    
        #A non empty detection will be split in more than 1 cell
        if len(detection) > 1:
            for i in range(1, len(detection)):
                
                # Extract box2d coordinates
                b = detection[i].find('box_2D:')
                e = detection[i].find('centroid:')
                box2D = detection[i][b:e]
                box2dList.append(box2D)
                match = re.findall('(\d+)\n', box2D)
                x = match[0]
                xList.append(x)
                y = match[1]
                yList.append(y)
                w = match[2]
                wList.append(w)
                h = match[3]
                hList.append(h)
                
                # Reduce information to avoid ambiguious match
                newdetection = detection[i][e:]
                
                # Extract height of the detection
                b = newdetection.find('height:')
                e = newdetection.find('confidence:')
                h = re.search('\d+\.\d+',newdetection[b:e]).group(0)
                heightList.append(float(h))
                
                # Reduce information to avoid ambiguious match
                newdetectionre = newdetection[e:]
                
                # Extract distance of the detection
                b = newdetectionre.find('distance:')
                e = newdetectionre.find('occluded:')
                d = re.search('\d+\.\d+',newdetectionre[b:e]).group(0)
                distanceList.append(float(d))
                
                # Extract timestamp
                time = msg.header.stamp
                detectTime.append(int(str(time)))
                
                #Extract seq number
                seq = msg.header.seq
                detectSeq.append(seq)
    
    
    
    #Creation of a dataframe of extracted information
    data = np.array([np.asarray(detectTime), np.asarray(detectSeq), 
            np.asarray(xList), np.asarray(yList), np.asarray(wList), 
            np.asarray(hList), np.asarray(heightList), 
            np.asarray(distanceList)]).T # The array need to be transposed
    detectionExtraction = pd.DataFrame(data=data)
    detectionExtraction.columns = ['detectionTimestamp', 'seq', 
            'detectionX', 'detectionY', 'detectionW', 'detectionH',
             'height', 'distance']
            
    return detectionExtraction
    
