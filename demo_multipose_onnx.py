#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
import tensorflow as tf


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


def run_inference(onnx_session, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # 前処理
    input_image = cv.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = input_image.astype('int32')  # int32へキャスト

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores = outputs[0]
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
    model_path = "onnx/movenet_multipose_lightning_1.onnx"
    input_size = 256

    onnx_session = onnxruntime.InferenceSession(model_path)

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
            onnx_session,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
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


def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints_list,
    scores_list,
    bbox_score_th,
    bbox_list,
):
    debug_image = copy.deepcopy(image)

    # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
    # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
    for keypoints, scores in zip(keypoints_list, scores_list):
        for index01, index02 in ((0, 1),  # Line：鼻 → 左目
                                 (0, 2),  # Line：鼻 → 右目
                                 (1, 3),  # Line：左目 → 左耳
                                 (2, 4),  # Line：右目 → 右耳
                                 (0, 5),  # Line：鼻 → 左肩
                                 (0, 6),  # Line：鼻 → 右肩
                                 (5, 6),  # Line：左肩 → 右肩
                                 (5, 7),  # Line：左肩 → 左肘
                                 (7, 9),  # Line：左肘 → 左手首
                                 (6, 8),  # Line：右肩 → 右肘
                                 (8, 10),  # Line：右肘 → 右手首
                                 (11, 12),  # Line：左股関節 → 右股関節
                                 (5, 11),  # Line：左肩 → 左股関節
                                 (11, 13),  # Line：左股関節 → 左ひざ
                                 (13, 15),  # Line：左ひざ → 左足首
                                 (6, 12),  # Line：右肩 → 右股関節
                                 (12, 14),  # Line：右股関節 → 右ひざ
                                 (14, 16),  # Line：右ひざ → 右足首
                                 ):
            if is_good_keypoints(index01, index02, keypoint_score_th, scores):
                draw_keypoints(debug_image, index01, index02, keypoints)

        # Circle：各点
        for keypoint, score in zip(keypoints, scores):
            if score > keypoint_score_th:
                cv.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
                cv.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    # バウンディングボックス
    for bbox in bbox_list:
        if bbox[4] > bbox_score_th:
            cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (255, 255, 255), 4)
            cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 0, 0), 2)

    # 処理時間
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
               cv.LINE_AA)
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
               cv.LINE_AA)

    return debug_image


def is_good_keypoints(index01, index02, keypoint_score_th, scores):
    return scores[index01] > keypoint_score_th and scores[
        index02] > keypoint_score_th


def draw_keypoints(debug_image, index01, index02, keypoints):
    point01 = keypoints[index01]
    point02 = keypoints[index02]
    cv.line(debug_image, point01, point02, (255, 255, 255), 4)
    cv.line(debug_image, point01, point02, (0, 0, 0), 2)


if __name__ == '__main__':
    main()