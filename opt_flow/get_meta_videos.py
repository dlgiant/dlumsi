import cv2
from os import listdir
from os.path import isfile, join

videos_dir = "videos"

def get_mp4_paths(parent=videos_dir):
    return [join(parent, f) for f in listdir(parent) if isfile(join(parent, f)) and f.endswith(".mp4")]

def get_fps(video_name):
    video = cv2.VideoCapture(video_name)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def get_total_frames(parent=videos_dir):
    total_frames = []
    for video in get_mp4_paths(videos_dir):
        cap = cv2.VideoCapture(video)
        total_frames.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frames

def get_time(fps, n_frames):
    return float(n_frames/fps)


def get_meta(parent=videos_dir):
    total_frames = get_total_frames(parent=videos_dir)
    return [(name, get_fps(name), frames, get_time(get_fps(name), frames)) for name, frames in zip(get_mp4_paths(parent=videos_dir), total_frames)]

def print_meta():
    for name, fps, frames, time in get_meta():
        print(f"\n\tNAME:{name}\n\tFPS:{fps}\n\tFRAMES:{frames}\n\tTIME:{time}\n")
        
