import os
from os.path import exists, join, basename, splitext
import gdown


def download_weights():

    yolo_pretrained_model_path = "detector/yolo/data/yolov3-spp.weights"
    if not exists(yolo_pretrained_model_path):
        try:
            os.mkdir("detector/yolo/data")
        except FileExistsError:
            pass
        gdown.download(
            "https://drive.google.com/uc?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC",
            yolo_pretrained_model_path,
        )

    pose_pretrained_model_path = "pretrained_models/fast_421_res152_256x192.pth"
    if not exists(pose_pretrained_model_path):
        gdown.download(
            "https://drive.google.com/uc?id=1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9",
            pose_pretrained_model_path,
        )

    track_pretrained_model_path = "trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
    if not exists(track_pretrained_model_path):
        try:
            os.mkdir("trackers/weights")
        except FileExistsError:
            pass
        gdown.download(
            "https://drive.google.com/uc?id=1TZ01_aX6WrJscTOOxKDdla2cP7SwxVzM",
            track_pretrained_model_path,
        )


if __name__ == "__main__":
    download_weights()
