import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from get_meta_videos import get_meta
import sys
from progress.bar import IncrementalBar, ShadyBar
import time
from tqdm import tqdm

all_frames = []
videos = []

skipper = int(sys.argv[1])


for meta in get_meta():
    video, fps, frames, seconds = meta
    minutes = round(seconds/60.0, 2)
    print(f"Processing: {video}\n\t\t{fps}\t{frames}\t{round(seconds/60.0, 2)}")
    cap = cv2.VideoCapture(video)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prvs_copy = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[    ...,1] = 255
    j = 0
    with tqdm(total=int(frames)) as pbar2:
        while(frames > j):
            ret, frame2 = cap.read()
            if j % skipper == 0:
                all_frames.append(frame2)
                k = cv2.waitKey(30) & 0xff
            if k == 27 :     
               break
            elif k == ord('s'):
               cv2.imwrite('opticalfb.png',frame2)
               cv2.imwrite('opticalhsv.png',rgb)
            j += 1
            pbar2.update(1)
    j = 0 

    for fr in all_frames:
        next = cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs_copy,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs_copy = next
        j += 1
    print("------------->Done)")

    res = np.concatenate([arr[np.newaxis] for arr in all_frames])
    print(f"Memory consumption estimate:\n\t\t{res.size}\t{res.shape}") 
    videos.append(res)
    print(f"\t\t{len(videos)}")

    cap.release()

cv2.destroyAllWindows()
