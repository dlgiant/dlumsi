from __future__ import print_function
import cv2 as cv
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(
            description='Subtracting background of Social interaction video')
parser.add_argument('--n', type=str, help='batch or random videos. (b)atch/{1..n}')

args = parser.parse_args()

back_sub_mog = cv.createBackgroundSubtractorMOG2()
back_sub_knn = cv.createBackgroundSubtractorKNN()


def video_to_array(path=""):
    cap = cv.VideoCapture(path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_heigth = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print("\n\t".join([path, " ".join([str(frame_count), str(frame_width), str(frame_heigth)])]))

    if not cap.isOpened:
        print("Unable to open video.")
        exit(0)

    buf = np.empty((frame_count, frame_heigth, frame_width, 3),
                   np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frame_count and ret):
        ret, buf[fc] = cap.read()
        if buf[fc] is None:
            break

        knn_mask = back_sub_knn.apply(buf[fc])
        mog_mask = back_sub_mog.apply(buf[fc])
        cv.rectangle(buf[fc], (10, 2), (100,20), (255, 255, 255), -1)
        cv.putText(buf[fc], str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('Frame', buf[fc])
        cv.imshow('MOG2', mog_mask)
        cv.imshow('KNN', knn_mask)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

        fc += 1
    cap.release()

    # data = np.asarray(buf)
    # np.save("{}{}".format(
    #         os.path.join("videos_npy",
    #                      path.split('/')[-1].split('.')[0]), ".npy"),
    #         data)


def get_videos_paths(directory="videos/"):
    all_videos = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            all_videos.append(os.path.join(dirname, filename))
    return all_videos


def get_all_videos():
    # pool = mp.Pool(30)
    for path in get_videos_paths():
        video_to_array(path)


def process_n_videos(n=1):
    import random as rd

    choices = rd.choices(get_videos_paths(), k=n)
    for choice in choices:
        video_to_array(choice)


if args.n[-1].lower() == 'b':
    videos = get_all_videos()
elif args.n.isnumeric():
    videos = process_n_videos(int(args.n))
