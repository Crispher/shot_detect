import cv2
import numpy as np
import os
import sys
from scipy.spatial.distance import cityblock

class shot_detector: 
    def __init__(self, video_path=None, min_duration=10, output_dir=None, thres = 1.5): 
        self.video_path = video_path
        self.min_duration = min_duration 
        self.output_dir = output_dir
        self.hist_size = 64             # how many bins for each R,G,B histogram
        self.absolute_threshold = thres # any transition must be no less than this threshold range from 0 to 3, the higher the more sensitive.
 
    def run(self, video_path=None):
        if video_path is not None:
            self.video_path = video_path    
        assert (self.video_path is not None), "you should must the video path!"

        self.shots = []
        cap = cv2.VideoCapture(self.video_path)
        hists = []
        frames = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            if self.output_dir is not None:
                frames.append(frame)
            # compute RGB histogram for each frame
            color_histgrams = [cv2.calcHist([frame], [c], None, [self.hist_size], [0,256]) \
                          for c in range(3)]
                          
            color_histgrams = np.array([chist/float(sum(chist)) for chist in color_histgrams])
            
            hists.append(color_histgrams.flatten())

        # manhattan distance of two consecutive histgrams
        scores = [cityblock(*h12) for h12 in zip(hists[:-1], hists[1:])]
        
        print("max diff:", max(scores), "min diff:", min(scores))
        
        # compute automatic threshold
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = self.absolute_threshold

        # decide shot boundaries
        prev_i = 0
        prev_score = scores[0]
        for i, score in enumerate(scores[1:]):
            if (score >= threshold) and (abs(score - prev_score) >= threshold/2.0):
                self.shots.append((prev_i, i+2))
                prev_i = i + 2
            prev_score = score
        video_length = len(hists)
        self.shots.append((prev_i, video_length))
        assert video_length>=self.min_duration, "duration error"

        self.merge_short_shots()
        
        # save key frames
        if self.output_dir is not None:
            for shot in self.shots:
                cv2.imwrite("%s/frame-%d.jpg" % (self.output_dir,shot[0]), frames[shot[0]])
            print("key frames written to %s" % self.output_dir)

    def merge_short_shots(self):
        # merge short shots
        while True:
            durations = [shot[1]-shot[0] for shot in self.shots]
            shortest = min(durations)
            # no need to merge
            if shortest >= self.min_duration:
                break
            idx = durations.index(shortest)
            left_half = self.shots[:idx]
            right_half = self.shots[idx+1:]
            shot = self.shots[idx]

            # can only merge left
            if idx == len(self.shots)-1:
                left = True                
            # can only merge right
            elif idx == 0:
                left = False                
            else:
                # otherwise merge the shorter one
                if durations[idx-1] < durations[idx+1]:
                    left = True
                else:
                    left = False
            if left:
                self.shots = left_half[:-1] + [(left_half[-1][0],shot[1])] + right_half
            else:
                self.shots = left_half + [(shot[0],right_half[0][1])] + right_half[1:]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ./shotdetect.py <video-path> <output_dir>")
        sys.exit()
    video_path = sys.argv[1]
    key_frames_dir = sys.argv[2]
    detector = shot_detector(video_path, output_dir=key_frames_dir, thres=1.5)
    detector.run()
    print(detector.shots)