Shot boundary detection
    Use histograms to detect shots of a video.
    Usage:
    >>> python shot_detect.py <video_path> <output_directory>
    The starting frame of each shot is stored to output_dir.
    
    Adjust sensitivity: tune the parameter thres(=1.5 by default) in range(0, 3). The higher the more sensitive.
    
    Environment:
    - Open CV 3.1.0, Python 3.5 binding.

This project extends the codes on github by @yu239.