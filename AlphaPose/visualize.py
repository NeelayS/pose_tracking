import cv2 as cv
import re


def visualize(frames_list, inp_video_path, output_video_path):

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
    ]

    fourcc = cv.VideoWriter_fourcc(*"avc1")

    video = cv.VideoCapture(inp_video_path)

    out_video = cv.VideoWriter(
        output_video_path,
        fourcc,
        video.get(cv.CAP_PROP_FPS),
        (int(video.get(3)), int(video.get(4))),
    )

    current_frame = 0
    ret = True

    person_color = {}
    color = 0

    while video.isOpened():
        ret, video_frame = video.read()

        if ret is False:
            break

        for frame in frames_list:

            if frame["idx"] not in person_color.keys():
                person_color[frame["idx"]] = colors[color]
                color += 1

            if int(frame["image_id"][:-4]) == current_frame:

                for i in range(len(frame["keypoints"]) // 3):
                    x, y = (
                        int(frame["keypoints"][3 * i]),
                        int(frame["keypoints"][3 * i + 1]),
                    )

                    try:
                        cv.circle(
                            video_frame,
                            (x, y),
                            radius=4,
                            color=person_color[frame["idx"]],
                            thickness=-1,
                        )
                    except:
                        cv.circle(
                            video_frame,
                            (x, y),
                            radius=4,
                            color=(255, 255, 255),
                            thickness=-1,
                        )

                keypoints = frame["keypoints"]

                coord_pairs = (
                    (0, 1),
                    (0, 2),
                    (1, 3),
                    (2, 4),
                    (5, 6),
                    (5, 7),
                    (6, 8),
                    (7, 9),
                    (8, 10),
                    (11, 12),
                    (11, 13),
                    (12, 14),
                    (13, 15),
                    (14, 16),
                )

                for pair in coord_pairs:
                    start_point = (
                        int(keypoints[3 * pair[0]]),
                        int(keypoints[3 * pair[0] + 1]),
                    )
                    end_point = (
                        int(keypoints[3 * pair[1]]),
                        int(keypoints[3 * pair[1] + 1]),
                    )

                    video_frame = cv.line(
                        video_frame,
                        start_point,
                        end_point,
                        person_color[frame["idx"]],
                        3,
                    )

                video_frame = cv.line(
                    video_frame,
                    (
                        int((keypoints[15] + keypoints[18]) / 2),
                        int((keypoints[16] + keypoints[19]) / 2),
                    ),
                    (
                        int((keypoints[33] + keypoints[36]) / 2),
                        int((keypoints[34] + keypoints[37]) / 2),
                    ),
                    person_color[frame["idx"]],
                    3,
                )

        out_video.write(video_frame)
        current_frame += 1

    video.release()
    out_video.release()
