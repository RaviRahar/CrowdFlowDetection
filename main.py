import numpy as np
import cv2 as cv
from CrowdFlowSegmentation import *

stride = 30

cap = cv.VideoCapture("UCF_CrowdsDataset/3687-18_70.mov")
# cap = cv.VideoCapture('UCF_CrowdsDataset/3687-18_70.mov')

no_of_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
ret, frame = cap.read()
if not ret:
    print("No frames grabbed!\n")

crowdFlow = CrowdFlowSegmentation(frame, no_of_frames, stride)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frames grabbed!\n")
        break
    # modified_frame = crowdFlow.to_dense(frame)
    # modified_frame = crowdFlow.to_sparse(frame)
    modified_frame = crowdFlow.to_streaklines(frame)
    # modified_frame = crowdFlow.to_similarity(modified_frame)
    # modified_frame = crowdFlow.to_watershed(modified_frame)
    cv.imshow("window", modified_frame)

    k = cv.waitKey(30) & 0xFF
    if k == 27 or k == ord("q"):
        break
    # to save screenshot of frame press 's'
    elif k == ord("s"):
        cv.imwrite("opticalfb.png", frame)
        cv.imwrite("opticalhsv.png", modified_frame)

cap.release()
cv.destroyAllWindows()
