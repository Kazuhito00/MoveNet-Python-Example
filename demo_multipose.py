#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

from pose_helper import draw_debug_multi_person


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--mirror', action='store_true')

    parser.add_argument("--keypoint_score", type=float, default=0.4)
    parser.add_argument("--bbox_score", type=float, default=0.2)

    args = parser.parse_args()

    return args


def run_inference(model, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # 前処理
    input_image = cv.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = tf.cast(input_image, dtype=tf.int32)  # int32へキャスト

    # 推論
    outputs = model(input_image)

    keypoints_with_scores = outputs['output_0'].numpy()
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # キーポイント、バウンディングボックス、スコア取り出し
    keypoints_list, scores_list = [], []
    bbox_list = []
    for keypoints_with_score in keypoints_with_scores:
        keypoints = []
        scores = []
        # キーポイント
        for index in range(17):
            keypoint_x = int(image_width *
                             keypoints_with_score[(index * 3) + 1])
            keypoint_y = int(image_height *
                             keypoints_with_score[(index * 3) + 0])
            score = keypoints_with_score[(index * 3) + 2]

            keypoints.append([keypoint_x, keypoint_y])
            scores.append(score)

        # バウンディングボックス
        bbox_ymin = int(image_height * keypoints_with_score[51])
        bbox_xmin = int(image_width * keypoints_with_score[52])
        bbox_ymax = int(image_height * keypoints_with_score[53])
        bbox_xmax = int(image_width * keypoints_with_score[54])
        bbox_score = keypoints_with_score[55]

        # 6人分のデータ格納用のリストに追加
        keypoints_list.append(keypoints)
        scores_list.append(scores)
        bbox_list.append(
            [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

    return keypoints_list, scores_list, bbox_list


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.file is not None:
        cap_device = args.file

    mirror = args.mirror
    keypoint_score_th = args.keypoint_score
    bbox_score_th = args.bbox_score

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    model_url = "https://tfhub.dev/google/movenet/multipose/lightning/1"
    input_size = 256

    module = tfhub.load(model_url)
    model = module.signatures['serving_default']

    cv.namedWindow('MoveNet(multipose) Demo', cv.WINDOW_NORMAL)

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            break
        if mirror:
            frame = cv.flip(frame, 1)  # ミラー表示
        debug_image = copy.deepcopy(frame)

        # 検出実施 ##############################################################
        keypoints_list, scores_list, bbox_list = run_inference(
            model,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug_multi_person(
            debug_image,
            elapsed_time,
            keypoint_score_th,
            keypoints_list,
            scores_list,
            bbox_score_th,
            bbox_list,
        )

        # キー処理(ESC：終了) ##################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MoveNet(multipose) Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()