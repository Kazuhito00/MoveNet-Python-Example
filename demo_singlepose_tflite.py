#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf

from pose_helper import draw_debug


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--mirror', action='store_true')

    parser.add_argument("--model_select", type=int, default=0)
    parser.add_argument("--keypoint_score", type=float, default=0.4)

    args = parser.parse_args()

    return args


def run_inference(interpreter, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # 前処理
    input_image = cv.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = tf.cast(input_image, dtype=tf.uint8)  # uint8へキャスト

    # 推論
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # キーポイント、スコア取り出し
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.file is not None:
        cap_device = args.file

    mirror = args.mirror
    model_select = args.model_select
    keypoint_score_th = args.keypoint_score

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    if model_select == 0:
        model_path = 'tflite/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite'
        input_size = 192
    elif model_select == 1:
        model_path = 'tflite/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite'
        input_size = 256
    elif model_select == 2:
        model_path = 'tflite/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite'
        input_size = 192
    elif model_select == 3:
        model_path = 'tflite/lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite'
        input_size = 256
    else:
        sys.exit(
            "*** model_select {} is invalid value. Please use 0-3. ***".format(
                model_select))

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    cv.namedWindow('MoveNet(singlepose) Demo', cv.WINDOW_NORMAL)
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
        keypoints, scores = run_inference(
            interpreter,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            keypoint_score_th,
            keypoints,
            scores,
        )

        # キー処理(ESC：終了) ##################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MoveNet(singlepose) Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
