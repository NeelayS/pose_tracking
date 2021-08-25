import pandas as pd
import os
from os.path import exists, join
import argparse
import re
from math import floor, ceil
from shutil import copyfile

parser = argparse.ArgumentParser(description="Script for trimming videos")

parser.add_argument(
    "--indir", type=str, required=True, help="Root directory of the videos"
)
parser.add_argument(
    "--outdir", type=str, required=True, help="Where the videos shpuld be stored"
)
parser.add_argument(
    "--csv",
    type=str,
    required=True,
    default="finalized_viewable_segmentUrls.csv",
    help="CSV file containing the names of videos and timestamps",
)

args = parser.parse_args()

if not exists(args.outdir):
    # print("The output directory doesn't exist. Creating it.")
    os.mkdir(args.outdir)

df = pd.read_csv(args.csv, header=None)
entries = df[0]
entries = [entry[57:] for entry in entries]

video_split_count = 0
previous_video_name = ""

num_videos_processed = 0

for entry in entries:

    category_name = re.search("(.*?)/", entry).group(1)
    video_name = re.search("/(.*?)#", entry).group(1)

    if video_name == previous_video_name:
        video_split_count += 1
    else:
        video_split_count = 0

    previous_video_name = video_name

    inp_video_path = join(args.indir, category_name, video_name)
    cut_video_path = join(
        args.outdir, category_name, str(video_split_count) + "_" + video_name
    )

    if not exists(inp_video_path):
        # print("This video doesn't exist. Moving to next one\n")
        continue

    else:
        num_videos_processed += 1

    if not exists(join(args.outdir, category_name)):
        os.mkdir(join(args.outdir, category_name))

    try:
        start_time = floor(float(re.search("#(.*?),", entry).group(1)))
        end_time = ceil(float(re.search(",(.*?)#", entry).group(1)))

    except:
        copyfile(inp_video_path, cut_video_path)
        continue

    start_min = start_time // 60
    start_sec = start_time - (60 * start_min)
    end_min = end_time // 60
    end_sec = end_time - (60 * end_min)

    command = f"ffmpeg -i {inp_video_path} -ss 00:{start_min}:{start_sec} -to 00:{end_min}:{end_sec} -c copy {cut_video_path} -loglevel quiet"

    os.system(command)

print(f"{num_videos_processed} videos were successfully processed")
