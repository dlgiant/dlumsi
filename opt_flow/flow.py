import numpy as np
import cv2
import time
import logging
import sys
from sklearn import random_projection

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger('cv_flow')

capture = cv2.VideoCapture('videos/MPHS3_top.mp4')

wait_for = float(sys.argv[1])

corner_detection_st = \
  dict( maxCorners = 100,
        qualityLevel = 0.01,
        minDistance = 20,
        blockSize = 1 )

lk_params = \
	dict( winSize  = (10,10),
        maxLevel = 10,
        criteria = (cv2.TERM_CRITERIA_EPS | 
			              cv2.TERM_CRITERIA_COUNT, 1, 0.3))

# Declaring transformer for random projection
transformer = random_projection.SparseRandomProjection()

# Take first frame and find corners in it
ret, old_frame = capture.read() # First frame
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) # converting to GRAY format 
prev_pts = cv2.goodFeaturesToTrack(old_gray, mask = None, **corner_detection_st) # Get corners
total_pts = len(prev_pts)

# Mask for drawing
mask = np.zeros_like(old_frame)

# Making random colors for the points
color = np.random.randint(0,255,(100,3))

print('Frame #rows: {} {} {}'.format(old_gray.shape, str(old_gray.ndim), str(old_gray.size)))

der_pts = np.array([old_frame])
j = 0 
while(1):
  ret, frame = capture.read()
  if not ret: break
 
  # convert to gray
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
  
  # calculate optical flow
  next_pts, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, prev_pts, None, **lk_params)
  
  #print(type(next_pts))
  #print(next_pts[status==1][:10].ndim)
  #print(next_pts[status==1].ndim)
  #print(next_pts[status==1].shape)

  #next_pts[status==1] = transformer.fit_transform(next_pts[status==1][:])
  #frame = transformer.fit_transform(frame)
  # Select good points
  
  good_new = next_pts[status==1]
  good_old = prev_pts[status==1]
  
  #der_pts += good_new
  #der_pts += good_old

  # draw the tracks
  for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    # draw trajectory for each point
    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    # draw circles in current frame
    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
  #print(mask.shape)
  #print(frame.shape)  
  
  #mask = transformer.fit_transform(mask)
  #frame = transformer.fit_transform(frame)
  
  # Blending frame and mask
  img = cv2.add(frame, mask)
  print(img.ndim, img.shape, img.size)
  der_pts = np.append(der_pts, img, axis=3)
  #img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
 
  #cv2.imshow('frame', img)
  k = cv2.waitKey(30) & 0xff
  if k == 27 or j == 50:
    break
  j += 1
  # Now update the previous frame and previous points
  old_gray = frame_gray.copy()
  prev_pts = good_new.reshape(-1,1,2)
  #print(mask.shape) 
  #time.sleep(wait_for)
    
print(der_pts.shape)
print(der_pts.ndim)
print(der_pts.size)

cv2.destroyAllWindows()
capture.release()
