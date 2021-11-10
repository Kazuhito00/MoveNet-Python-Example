import copy

import cv2 as cv

def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints,
    scores,
):
    debug_image = copy.deepcopy(image)

    # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
    # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
    for index01, index02 in ((0, 1), # Line：鼻 → 左目
                             (0, 2), # Line：鼻 → 右目
                             (1, 3), # Line：左目 → 左耳
                             (2, 4), # Line：右目 → 右耳
                             (0, 5), # Line：鼻 → 左肩
                             (0, 6), # Line：鼻 → 右肩
                             (5, 6), # Line：左肩 → 右肩
                             (5, 7), # Line：左肩 → 左肘
                             (7, 9), # Line：左肘 → 左手首
                             (6, 8), # Line：右肩 → 右肘
                             (8, 10), # Line：右肘 → 右手首
                             (11, 12), # Line：左股関節 → 右股関節
                             (5, 11), # Line：左肩 → 左股関節
                             (11, 13), # Line：左股関節 → 左ひざ
                             (13, 15), # Line：左ひざ → 左足首
                             (6, 12), # Line：右肩 → 右股関節
                             (12, 14), # Line：右股関節 → 右ひざ
                             (14, 16), # Line：右ひざ → 右足首
                             ):
        if is_good_keypoints(index01, index02, keypoint_score_th, scores):
            draw_keypoints(debug_image, index01, index02, keypoints)

    # Circle：各点
    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
            cv.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

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


def draw_debug_multi_person(
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
        for index01, index02 in ((0, 1), # Line：鼻 → 左目
                                 (0, 2), # Line：鼻 → 右目
                                 (1, 3), # Line：左目 → 左耳
                                 (2, 4), # Line：右目 → 右耳
                                 (0, 5), # Line：鼻 → 左肩
                                 (0, 6), # Line：鼻 → 右肩
                                 (5, 6), # Line：左肩 → 右肩
                                 (5, 7), # Line：左肩 → 左肘
                                 (7, 9), # Line：左肘 → 左手首
                                 (6, 8), # Line：右肩 → 右肘
                                 (8, 10), # Line：右肘 → 右手首
                                 (11, 12), # Line：左股関節 → 右股関節
                                 (5, 11), # Line：左肩 → 左股関節
                                 (11, 13), # Line：左股関節 → 左ひざ
                                 (13, 15), # Line：左ひざ → 左足首
                                 (6, 12), # Line：右肩 → 右股関節
                                 (12, 14), # Line：右股関節 → 右ひざ
                                 (14, 16), # Line：右ひざ → 右足首
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