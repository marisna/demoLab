import cv2 as cv 
import numpy as np
from numpy import linalg as LA
import random

def splitBodyPoints(currentKeypoints):
    keyTopPoints = [0,1,2,3,4,5,6,7,15,16,17,18]
    topPoints = [[int(values) for values in currentKeypoints[key]] for key in keyTopPoints if currentKeypoints[key] !=[0, 0]]
    bottomPoints = [[int(values) for values in currentKeypoints[idx]] for idx, v in enumerate(currentKeypoints) if idx not in keyTopPoints and currentKeypoints[idx] !=[0, 0]]
    return topPoints, bottomPoints

def getMaxMinBodyPt(points):
    idx, yVals = [],[]
    for enum, coords in enumerate(points):
        if coords != [0.0, 0.0]:
            yVals.append(coords[1])
            idx.append(enum)        
    top, bottom = np.argmin(np.array(yVals)), np.argmax(np.array(yVals))
    topCoords = tuple([int(points[idx[top]][0]), int(points[idx[top]][1])])
    bottomCoords = tuple([int(points[idx[bottom]][0]), int(points[idx[bottom]][1])])
    return topCoords, bottomCoords

def getCoordsMean(*pointsLists):
    mean = {}
    for enum, list_ in enumerate(pointsLists):
        X = np.array([x for x,y in list_])
        Y = np.array([y for x,y in list_])
        xMean, yMean = int(np.mean(X)), int(np.mean(Y))   
        mean['%s' %(enum+1)]= (xMean, yMean)       
    return mean
    
## A COMPACTER + METTRE AU PROPRE
def balancePointsQuantity(topPoints,bottomPoints,thr):
    topPoints_randGen = []
    if len(topPoints) < int(len(bottomPoints)*thr) :
        diff = len(bottomPoints)-int(len(topPoints))

        X = [topPoints[idx][0] for idx, x in enumerate(topPoints)]
        Y = [topPoints[idx][1] for idx, y in enumerate(topPoints)]

        topPoints_xmin, topPoints_ymin = min(X), min(Y)
        topPoints_xmax, topPoints_ymax = max(X), max(Y)      

        try:
            topPoints_randGen_x = [random.randrange(topPoints_xmin, topPoints_xmax) for _ in range(0,diff)]
        except:
            topPoints_randGen_x = [topPoints_xmax for _ in range(0,diff)]
        try:
            topPoints_randGen_y = [random.randrange(topPoints_ymin, topPoints_ymax) for _ in range(0,diff)]
        except ValueError:
            topPoints_randGen_y = [topPoints_ymin for _ in range(0,diff)]

        merge_xy = [[topPoints_randGen_x[idx], topPoints_randGen_y[idx]] for idx, values in enumerate(topPoints_randGen_x)]

        topPoints += merge_xy

    elif len(bottomPoints) < int(len(topPoints)*thr) :

        diff = len(topPoints)-len(bottomPoints)
        
        X = [bottomPoints[idx][0] for idx, x in enumerate(bottomPoints)]
        Y = [bottomPoints[idx][1] for idx, y in enumerate(bottomPoints)]

        bottomPoints_xmin, bottomPoints_ymin = min(X), min(Y)

        bottomPoints_xmax, bottomPoints_ymax = max(X),max(Y)

        try :
            bottomPoints_randGen_x = [random.randrange(bottomPoints_xmin, bottomPoints_xmax) for _ in range(0,diff)]
        except:
            bottomPoints_randGen_x = [bottomPoints_xmax for _ in range (0,diff)]
        try :
            bottomPoints_randGen_y = [random.randrange(bottomPoints_ymin, bottomPoints_ymax) for _ in range(0,diff)]
        except ValueError:
            bottomPoints_randGen_y = [bottomPoints_ymin for _ in range(0,diff)]

        merge_xy = [[bottomPoints_randGen_x[idx], bottomPoints_randGen_y[idx]] for idx, values in enumerate(bottomPoints_randGen_x)]
  
        bottomPoints += merge_xy

    return topPoints,bottomPoints

def getHumanVect(topPoints,bottomPoints, thr):    
    topPoints_smaller = len(topPoints) < int(len(bottomPoints)*thr)
    bottomPoints_smaller = len(bottomPoints) < int(len(topPoints)*thr)
    
    if (topPoints_smaller) | (bottomPoints_smaller) :
        topPoints_,bottomPoints_ = balancePointsQuantity(topPoints,bottomPoints, thr)                
        mean = getCoordsMean(topPoints_,bottomPoints_)
        p1, p2 = mean['1'],  mean['2']
    else:  
        mean = getCoordsMean(topPoints,bottomPoints)
        p1, p2 = mean['1'],  mean['2']
    humanVect = np.array(p1) - np.array(p2)
    return humanVect, p1, p2

def getRefVect(frame):
    p1 = (int(frame.shape[1]*0.20),280)
    p2 = (int(frame.shape[1]*0.80),280)
    refVect = np.array(p1) - np.array(p2)
    return refVect

def getAngles(humanVect, refVect):
    radian = np.math.atan2(np.linalg.det([humanVect, refVect]), np.dot(humanVect, refVect))
    angle = np.degrees(radian)
    return angle

def computeAngle(flow, frame, currentKeypoints, upper_body_pts,lower_body_pts, counterbalance_pts_thr):
    humanVect, humanTopPoint, humanBottomPoint = getHumanVect(upper_body_pts,lower_body_pts, counterbalance_pts_thr)         
    refVect = getRefVect(frame)
    angle = int(LA.norm(getAngles(humanVect, refVect)))
    return angle, humanTopPoint, humanBottomPoint

def getPreviousCoordsForMissingPoints(currentKeypoints, previousKeypoints):
    missingPoints = []
    missingPointsIndexes = []
    for articulationIndex, coords in enumerate(currentKeypoints):
        if np.array([coords]).sum() == 0 and np.array([previousKeypoints[articulationIndex]]).sum() != 0.0 :
            missingPoints.append(previousKeypoints[articulationIndex])
            missingPointsIndexes.append(articulationIndex) 
    missingPoints = np.array(missingPoints, dtype = np.float32)
    return missingPoints, missingPointsIndexes