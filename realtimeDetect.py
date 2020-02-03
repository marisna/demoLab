import os 
import glob
import cv2 as cv 
import sys
# from sys import platform
import argparse
import time
import vectorizer
import numpy as np
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from numpy import linalg as LA
import random
from scipy.spatial.distance import cdist 
import time

#THIS SCRIPT ASSUME THAT YOU HAVE CLONED AND COMPILED OPENPOSE REPO FOR PYTHON
# Following path args refers to the original openpose project dir

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', required = True, help = 'custom path of openpose/build/examples/tutorial_api_python dir ')
parser.add_argument('--thr', default = 0.8, help ='confidence score beyond which openpose detections will be accepted')
parser.add_argument('--video', required = True, help = ' video source path - int 0 if from webcam ')
parser.add_argument('--flip', default = False, help= 'flip frames of 180° horizontally ')

args = parser.parse_args()

# /!\ S'assurer que les .dll contenues dans le dossier /bin sont copiées dans le dir x64/Release/ avec l'OpenposeDemo.exe

sys.path.append(args.dir_path + '/../../python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + args.dir_path + '/../../x64/Release;' +  args.dir_path + '/../../bin;'

import pyopenpose as op

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = args.dir_path + "'/../../../models"
params["render_threshold"] = args.thr

# START OPENPOSE
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
# LAUNCH REAL TIME VIDEO FLOW
if args.video == '0' :
    flow = WebcamVideoStream(src=0).start()
else:
    flow = WebcamVideoStream(src=args.video).start()
fps = FPS().start()

previousKeypointsInit = False 
previousFrameInit = False

lk_params = dict( winSize  = (21,21),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1, 0.01))

#counterbalanceThr is a thr value useful to balance points - EXPLAIN BETTER
counterbalanceThr = 0.75
defaultAngle = 80
angularConfInterval = [value for value in range(defaultAngle-50, defaultAngle+50, 1)]

#COLOR LIST FOR FUTURE MATCHING BY COMMON INDEX - handle 10 people 
color = dict(zip(range(1,11), [tuple(random.randrange(0,255) for _ in range(0,3)) for _ in range(0,10)]))
frameNumber = 0

startTime = time.time()

while not cv.waitKey(1) & 0xFF == ord('q'):
    frame = flow.read()
    if args.flip == True:
        frame = cv.flip(frame, 0)

    currentGrayedFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)      
    #FILL OPENPOSE NETS WITH CURRENT FRAME 
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])    
    #DRAW REFERENCE VECTOR IN THE WIDTH OF THE FRAME
    # cv.line(frame, (int(frame.shape[1]*0.20),frame.shape[0]-20),(int(frame.shape[1]*0.80),frame.shape[0]-20), (255,255,255), 3)    
    # cv.putText(frame, 'Ref vector', (int(frame.shape[1]*0.40), frame.shape[0]-30),  cv.FONT_HERSHEY_SIMPLEX ,0.75, (255,255,255),2)
    if datum.poseKeypoints.shape != ():       
        currentKeypoints = [[[triple[idx] for idx, elem in enumerate(triple) if idx != 2] for triple in points] for points in datum.poseKeypoints]   
        currentSortedPoints = []  

        # GRAB FIRST FRAME WHEN OPENPOSE POINTS ARE DETECTED JUST TO FEED VARS WHOM SETUP IS NEEDED FOR DETECTION
        if previousKeypointsInit == previousFrameInit == False :
            previousKeypoints = currentKeypoints
            previousGrayedFrame = currentGrayedFrame
            previousKeypointsInit = previousFrameInit = True             
            fps.update()   
            frameNumber += 1
            continue   
        # ONCE 'SETUP' IS DONE, DETECTION CAN START    
        else:
            # IF A PEOPLE APPEAR, PREV PTS NEED TO BE RECOMPUTED
            if len(currentKeypoints) <= len(previousKeypoints):

                # COMPUTE OPTICAL FLOW PREDS FROM PREVIOUS PTS STORED 
                keyToReorderPoints = {} 
                track = []            
                for prevKeypoints in previousKeypoints:
                    prevKeypointsArr = np.array(prevKeypoints, dtype = np.float32).reshape(len(prevKeypoints), 1, 2)
                    predsTrack, st, err = cv.calcOpticalFlowPyrLK(previousGrayedFrame, currentGrayedFrame, prevKeypointsArr, None,  **lk_params)             
                    # for enumPred, pred in enumerate(predsTrack):
                    #     # print(pred)
                    #     if st[enumPred] ==1:
                    #         cv.circle(frame, (pred[0][0], pred[0][1]),6, (0,250,250), -1)                    

                    track.append(predsTrack)

                # GET SIMILARIEST COORDS BETWEEN PREDS FROM OPTICAL FLOW AND CURRENT PTS TO SORTED THE FLOW AND ASSOCIATE ID TO PEOPLE   
                for enumRaw, individualKeypoints in enumerate(currentKeypoints):
                    matchId = {}                           
                    similarityList  =  []
                    # JUST WRITE ON IMAGE THE OPENPOSE IDX OF THE BODY DETECTED - useful for visualization of the tracking job 
                    # up, down = vectorizer.getMaxMinBodyPt(individualKeypoints)             
                    # cv.putText(frame, "%s" % str(enumRaw+1), (up[0]+10, up[1]+10), cv.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), thickness= 3 )
                    # cv.putText(frame, "%s" % str(enumRaw+1), (up[0]+10, up[1]+10), cv.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,200), thickness= 1)

                    for enum, pts in enumerate(track):
                        similarityScores = []
                        for enumPts, val in enumerate(pts):
                            if (np.sum(np.array(val)) != 0) | (np.sum(np.array(individualKeypoints[enumPts]) != 0)):
                                similarityScores.append(cdist(val, [individualKeypoints[enumPts]], 'minkowski')) 
                        score = np.sum(np.array(similarityScores))
                        similarityList.append(score)                    
                    pointerPrevKpts = np.argmin(np.array(similarityList))
                    keyToReorderPoints[pointerPrevKpts] = enumRaw

                #REORDONNATE CURRENT KP BASED ON ORDER FOUND FOR PREV KP      
                
                for k, v in sorted(keyToReorderPoints.items()):
                    currentSortedPoints.append(currentKeypoints[v])

                for enumSortedPts, indivSortedPts in enumerate(currentSortedPoints):
                    # for coords in indivSortedPts:
                    #     if np.array(coords).sum() > 0: # and len(coords) > 1 :
                    #         cv.circle(frame, (int(coords[0]), int(coords[1])), 4, color[enumSortedPts+1], -1)

                    upperBodyPts, lowerBodyPts = vectorizer.splitBodyPoints(indivSortedPts)

                    if (len(upperBodyPts) in range(0,2)) | (len(lowerBodyPts) in range(0,2)):
                        fps.update()
                        continue

                    # COMPUTE ANGLE                   
                    humanVect, humanTopPoint, humanBottomPoint = vectorizer.getHumanVect(upperBodyPts,lowerBodyPts, counterbalanceThr)    
                    refVect = vectorizer.getRefVect(frame)
                    angle = int(LA.norm(vectorizer.getAngles(humanVect, refVect)))

                    cv.line(frame,humanTopPoint, humanBottomPoint, (255,255,255), 4)
                    cv.line(frame,humanTopPoint, humanBottomPoint, (64,64,64), 2)
                    # cv.putText(frame, "%s" % str(enumSortedPts+1), (humanTopPoint[0]+5, humanTopPoint[1]+5), cv.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), thickness= 5 )
                    # cv.putText(frame, "%s" % str(enumSortedPts+1), (humanTopPoint[0]+5, humanTopPoint[1]+5), cv.FONT_HERSHEY_SIMPLEX , 1, color[enumSortedPts+1], thickness= 2 )
                        
                    if angle not in angularConfInterval : #and setup.is_in_bed(humanTopPoint,humanBottomPoint, x, y, w, h) is False :
                        cv.putText(frame, 'FALL SUSPECTED!', (150,50), cv.FONT_HERSHEY_SIMPLEX ,1, (0,0,255),6)            
                previousKeypoints = currentSortedPoints
            # CASE OF A PERSON APPEAR NEED TO STORE NEW PREV KEYPTS AS REF  
            else:
                previousKeypoints = currentKeypoints

    frameNumber +=1
    fps.update()

    cv.putText(frame, 'frame : %s' %frameNumber, (30, frame.shape[0]- 50), cv.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), 1)
    cv.putText(frame, 'FPS : %s' %(int(frameNumber/(time.time()-startTime))), (frame.shape[1]-150, 50), cv.FONT_HERSHEY_SIMPLEX , 0.75, (255,255,255), 4)
    cv.putText(frame, 'FPS : %s' %(int(frameNumber/(time.time()-startTime))), (frame.shape[1]-150, 50), cv.FONT_HERSHEY_SIMPLEX , 0.75, (0,0,0), 1)
    cv.putText(frame, '%s person detected' %len(currentKeypoints), (frame.shape[1]- 200, frame.shape[0] - 50), cv.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), 1)
    cv.imshow("Fall Detection with Openpose", frame)
cv.destroyAllWindows()


