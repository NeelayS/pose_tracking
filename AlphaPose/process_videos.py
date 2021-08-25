import os
from os.path import exists, join
import argparse
import json

from visualize import visualize


def generate_tracklets(results_list):
    tracklets = {}

    for frame in results_list:
        person = frame["idx"]

        if not person in tracklets.keys():
            tracklets[person] = {}

        keypoints = frame["keypoints"]
        coordinate = [
            (keypoints[3 * i], keypoints[3 * i + 1]) for i in range(len(keypoints) // 3)
        ]
        confidence_score = [keypoints[3 * i + 2] for i in range(len(keypoints) // 3)]
        overall_score = frame["score"]

        if not "coordinates" in tracklets[person].keys():
            tracklets[person]["coordinates"] = []
            tracklets[person]["confidence_scores"] = []
            tracklets[person]["overall_scores"] = []

        tracklets[person]["coordinates"].append(coordinate)
        tracklets[person]["confidence_scores"].append(confidence_score)
        tracklets[person]["overall_scores"].append(overall_score)

        for person in tracklets.keys():
            tracklets[person]["avg_overall_score"] = sum(
                tracklets[person]["overall_scores"]
            ) / len(tracklets[person]["overall_scores"])

    return tracklets


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to generate tracklets from all videos in a directory"
    )

    parser.add_argument(
        "--indir",
        type=str,
        required=True,
        help="Should point to directory containing all the trimmed videos",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Location where all the output files should be saved",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=2.75,
        help="Threshold score for filtering videos",
    )

    parser.add_argument(
        "--visualize",
        type=bool,
        default=True,
        help="Whether to the visualize the tracklets for videos which pass the filters",
    )

    parser.add_argument(
        "--min_time_fraction",
        type=float,
        default=0.2,
        help="Min fraction of time the person must be present in the video to be considered",
    )

    parser.add_argument(
        "--n_people",
        type=int,
        default=5,
        help="Top k people to be stored for the videos on which performance is suitable",
    )

    parser.add_argument(
        "--n_cat_videos",
        type=int,
        default=5,
        help="No. of videos of each category required",
    )

    parser.add_argument(
        "--detbatch", type=int, default=5, help="detection batch size PER GPU"
    )

    parser.add_argument(
        "--posebatch",
        type=int,
        default=10,
        help="pose estimation maximum batch size PER GPU",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        dest="gpus",
        default="0",
        help="choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)",
    )
    parser.add_argument(
        "--qsize",
        type=int,
        dest="qsize",
        default=32,
        help="the length of result buffer, where reducing it will lower requirement of cpu memory",
    )

    args = parser.parse_args()

    ckpt = "pretrained_models/fast_421_res152_256x192.pth"
    cfg = "configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"

    if not exists(args.outdir):
        #  print("The output directory doesn't exist. Creating it.")
        for name in ["AP_results", "tracklets", "videos"]:
            os.makedirs(join(args.outdir, name), exist_ok=True)
        os.mkdir(join(args.outdir, "AP_results", "all"))
        os.mkdir(join(args.outdir, "AP_results", "filtered"))

    for category in os.listdir(args.indir):

        if not exists(join(args.outdir, "tracklets", category)):
            os.mkdir(join(args.outdir, "tracklets", category))

        if not exists(join(args.outdir, "videos", category)):
            os.mkdir(join(args.outdir, "videos", category))

        if not exists(join(args.outdir, "AP_results", "all", category)):
            os.makedirs(join(args.outdir, "AP_results", "all", category))

        if not exists(join(args.outdir, "AP_results", "filtered", category)):
            os.makedirs(join(args.outdir, "AP_results", "filtered", category))

        category_count = 0

        for video in os.listdir(join(args.indir, category)):

            if exists(
                join(args.outdir, "AP_results", "all", category, video[:-4] + ".json")
            ):
                print(
                    f"The results for video {category}/{video[:-4]} exist already. Moving to next\n"
                )
                continue

            if category_count == args.n_cat_videos:
                break

            video_path = join(args.indir, category, video)
            results_path = join(args.outdir, "AP_results", "all", category)

            print(f"Processing video {category}/{video}")

            command = f"python scripts/demo_inference.py --checkpoint {ckpt} --cfg {cfg} --video {video_path} --outdir {results_path} --gpus {args.gpus} --detbatch {args.detbatch} --posebatch {args.posebatch} --qsize {args.qsize} --pose_track"

            os.system(command)

            try:
                file_handle = open(join(results_path, "alphapose-results.json"))
            except:
                print("There was a problem with processing this video. Moving to next")
                continue

            results_list = json.load(file_handle)
            file_handle.close()

            tracklets = generate_tracklets(results_list)

            n_frames = len(results_list)
            min_frames = int(n_frames * args.min_time_fraction)

            filtered_persons = {
                person: tracklets[person]["avg_overall_score"]
                for person in tracklets.keys()
                if len(tracklets[person]["overall_scores"]) > min_frames
            }

            if not filtered_persons or len(filtered_persons.keys()) < 2:
                print(
                    "This video is not suitable to be a part of the dataset. Moving to next\n"
                )

                os.rename(
                    join(results_path, "alphapose-results.json"),
                    join(results_path, video[:-4] + ".json"),
                )

                continue

            sorted_filtered_persons = {
                person: score
                for person, score in sorted(
                    filtered_persons.items(), key=lambda item: item[1], reverse=True
                )
            }

            is_video_suitable = 1

            for person in list(sorted_filtered_persons.keys())[:2]:
                if not sorted_filtered_persons[person] >= args.threshold:
                    is_video_suitable = 0
                    break

            if is_video_suitable:

                print("This video is suitable to be a part of the dataset\n")

                try:
                    person_ids = list(sorted_filtered_persons.keys())[: args.n_people]
                except:
                    person_ids = sorted_filtered_persons.keys()

                filtered_tracklets = {
                    person: tracklet
                    for person, tracklet in tracklets.items()
                    if person in person_ids
                }
                filtered_results_list = [
                    person for person in results_list if person["idx"] in person_ids
                ]

                tracklets_json = json.dumps(filtered_tracklets)
                tracklets_file_handle = open(
                    join(
                        args.outdir,
                        "tracklets",
                        category,
                        "tracklets_" + video[:-4] + ".json",
                    ),
                    "w",
                )
                tracklets_file_handle.write(tracklets_json)
                tracklets_file_handle.close()

                results_json = json.dumps(filtered_results_list)
                results_file_handle = open(
                    join(
                        args.outdir,
                        "AP_results",
                        "filtered",
                        category,
                        "filtered_" + video[:-4] + ".json",
                    ),
                    "w",
                )
                results_file_handle.write(results_json)
                results_file_handle.close()

                if args.visualize:
                    visualize(
                        filtered_results_list,
                        video_path,
                        join(args.outdir, "videos", category, video[:-4] + ".mp4"),
                    )

                category_count += 1

            else:
                print(
                    "This video is not suitable to be a part of the dataset. Moving to next\n"
                )

            os.rename(
                join(results_path, "alphapose-results.json"),
                join(results_path, video[:-4] + ".json"),
            )
