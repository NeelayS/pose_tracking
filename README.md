# pose_tracking
Pose tracking for a collection of videos using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)


## Pre-requisites

* CUDA 10.1

* A directory containing videos for tracking 

## Steps -

1. `cd` into `AlphaPose`
2. Create conda environment using the `ap-env.yml` file and activate
3. `export PATH=/usr/local/cuda/bin/:$PATH` <br>
   `export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH`
4. `sudo apt-get install libyaml-dev`
5. `python setup.py build develop` (<b>Hack:</b> If AlphaPose doesn't build properly after executing this command, doing it again works)
6. Run the `get_weights.py` script to download pre-trained model weights
7. Run `process_videos.py` (<b>Note:</b> All paths specified as arguments to the script must be absolute right from `root(/)` for OpenCV to work)

<br>

### `process_videos.py`

```
usage: process_videos.py [-h] --indir INDIR --outdir OUTDIR [--threshold THRESHOLD]
            [--visualize VISUALIZE] [--min_time_fraction MIN_TIME_FRACTION]
            [--n_people N_PEOPLE] [--n_cat_videos N_CAT_VIDEOS]
            [--detbatch DETBATCH] [--posebatch POSEBATCH] [--gpus GPUS]
            [--qsize QSIZE]

Script to generate tracklets from all videos in a directory

optional arguments:
  -h, --help            show this help message and exit
  --indir INDIR         Should point to directory containing all the trimmed
                        videos
  --outdir OUTDIR       Location where all the output files should be saved
  --threshold THRESHOLD
                        Threshold score for filtering videos
  --visualize VISUALIZE
                        Whether to the visualize the tracklets for videos
                        which pass the filters
  --min_time_fraction MIN_TIME_FRACTION
                        Min fraction of time the person must be present in the
                        video to be considered
  --n_people N_PEOPLE   Top k people to be stored for the videos on which
                        performance is suitable
  --detbatch DETBATCH   detection batch size PER GPU
  --posebatch POSEBATCH
                        pose estimation maximum batch size PER GPU
  --gpus GPUS           choose which cuda device to use by index and input
                        comma to use multi gpus, e.g. 0,1,2,3. (input -1 for
                        cpu only) 
```

Example arguments for the `process_videos.py` script -

`python process_videos.py --indir /home/user/trimmed_videos --outdir /home/user/results --threshold 3.0 --visualize True --min_time_fraction 0.25 --n_people 5 --gpus 0,1,2,3 --posebatch 10 --detbatch 10`

<br>

<b>Note:</b> OpenCV is used for visualization purposes. If you wish to make use of the visualization functioanlity, installing OpenCV from source on your system is recommended.
You may refer to the following blogs for instructions on how to install OpenCV from source on Ubuntu -
- [https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
- [https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/)


<br><br>
