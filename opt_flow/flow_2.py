import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

videos_dir = "videos"

all_videos = [join(videos_dir, f) for f in listdir(videos_dir) if isfile(join(videos_dir, f))]

frames = []
videos = []

for video in all_videos:
    cap = cv2.VideoCapture(video)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[    ...,1] = 255
    #print(frame1.ndim, frame1.shape, hsv.ndim, hsv.shape)

    #  res = np.array(frame1)
    j = 0

    while(1):
        ret, frame2 = cap.read()
        frames.append(frame2)
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        #cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27 or j == 1000:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next
        j += 1

    res = np.concatenate([arr[np.newaxis] for arr in frames])
    videos.append(res)
    print(res.shape)

    cap.release()

cv2.destroyAllWindows()
