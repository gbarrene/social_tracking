''' Extraction of the timestamp, the seq number, the detection 
    coordinate of the 2D box the height and the distance from the Kinect
    of the detection '''

# Basic import
import rosbag
import numpy as np
import pandas as pd
from cv_bridge import CvBridge, CvBridgeError
import re

def track_extraction(bagPath):
    
    # Create bag to process
    bag = rosbag.Bag(bagPath)
    
    # create bridge between CV and ROS
    br = CvBridge()
    
    # Creation of empty list thta will be fill by extracted information and then use to create a dataframe of useful info
    detectTime = []
    detectSeq = []
    idList = []
    box2dList = []
    xList = []
    yList = []
    wList = []
    hList = []
    heightList = []
    distanceList = []
    
    for topic, msg, t in bag.read_messages(topics=['/tracker/tracks']):
        begin = str(msg).find('tracks:')
        tracks = str(msg)[begin:]
    
        track = tracks.split("- ")
    
        if len(track) > 1:
            for i in range(1, len(track)):
                
                # Extract the id
                match = re.search('id: \d+', track[i])
                if match:
                    idList.append(int(re.search('\d+', match.group(0)).group(0)))
                    
                # Extract the height of the person detected 
                match = re.search('height: \d+(\.\d+)?', track[i])
                if match:
                    heightList.append(float(re.search('\d+(\.\d+)?', match.group(0)).group(0)))
                    
                # Extract the distance of the person detected from the 
                # Kinect that send the dectetion msg
                match = re.search('distance: \d+(\.\d+)?', track[i])
                if match:
                    distanceList.append(float(re.search('\d+(\.\d+)?', match.group(0)).group(0)))
                
                # Extract box2d coordinates
                b = track[i].find('box_2D:')
                box2D = track[i][b:]
                box2dList.append(box2D)
                match = re.findall('([a-zA-Z]+: )(-?\d+)', box2D)
                x = match[0][1]
                xList.append(int(x))
                y = match[1][1]
                yList.append(int(y))
                w = match[2][1]
                wList.append(int(w))
                h = match[3][1]
                hList.append(int(h))
                
                
                # Extract timestamp
                time = msg.header.stamp
                detectTime.append(int(str(time)))
                
                #Extract seq number
                seq = msg.header.seq
                detectSeq.append(int(seq))
    
    #Creation of a dataframe of extracted information
    data = np.array([np.asarray(detectTime, dtype=int), np.asarray(detectSeq, dtype=int), np.asarray(idList, dtype=int),
            np.asarray(xList, dtype=int), np.asarray(yList, dtype=int), np.asarray(wList, dtype=int), 
            np.asarray(hList, dtype=int), np.asarray(heightList), 
            np.asarray(distanceList)]).T # The array need to be transposed
    trackExtraction = pd.DataFrame(data=data)
    trackExtraction.columns = ['trackTimestamp', 'traSeq', 'trackId',
            'trackX', 'trackY', 'trackW', 'trackH', 'height', 'distance']
    
    return trackExtraction
