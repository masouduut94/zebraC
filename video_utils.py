import cv2
import numpy as np
from vidaug import augmentors as va
from pathlib import Path


#################################### Video Related utilities

def get_augmentation():
    """
    Use augmentation in pipeline if you like to
    """
    sometimes = lambda aug: va.Sometimes(0.5, aug)
    often = lambda aug:va.Sometimes(0.8, aug)

    vid_augment = va.Sequential([ 
        sometimes(va.RandomRotate(degrees=10)),
        often(va.HorizontalFlip()) 
    ])
    return vid_augment
    
def load_video(filename: str, n_frames=30, use_aug=False):
    """
    Samples the video to `n_frames` number of frames.
    
    Args:
        filename (str): 
        n_frames (int): number of frames to pick from video.
        use_aug (bool): whether to augment the video frames or not.
    
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened() or not Path(filename).is_file():
        print("Check the video file path", filename)
        return None
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_numbers = np.linspace(0, total-1, num=n_frames, dtype=int).tolist()
    frames = []
    
    for f_no in f_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
        status, frame = cap.read()
        frame = cv2.resize(frame, (320, 240))
        frames.append(frame)
        
    if use_aug:
        aug = get_augmentation()
        frames = aug(frames)
        
    return np.array(frames)