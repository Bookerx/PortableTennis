from src.human_detection import HumanDetector
import cv2
import config as config
from utils.generate_json import JsonGenerator
from utils.filter_result import ResultFilterer
from utils.visualize import Visualizer
from src.court.court_detector import CourtDetector
from src.court.top_view import TopViewProcessor
import numpy as np

detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg = config.detector_cfg, \
                                                                                           config.detector_weight, config.pose_weight, config.pose_model_cfg, config.pose_data_cfg
write_json = config.write_json
filter_criterion = config.filter_criterion


class FrameProcessor:
    def __init__(self):
        self.HP = HumanDetector(detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg,
                                config.ball_weight, config.ball_model_cfg, config.ball_data_cfg)
        self.write_json = write_json
        self.court_detector = CourtDetector()
        self.top_view = TopViewProcessor(self.court_detector.court_reference.net)

        if write_json:
            if not config.json_path:
                try:
                    json_path = config.input_src[:-len(config.input_src.split(".")[-1]) - 1] + ".json"
                except:
                    json_path = "result.json"
            else:
                json_path = ""
            self.Json = JsonGenerator(json_path)
        self.filter = ResultFilterer(filter_criterion)
        self.visualizer = Visualizer(self.HP.estimator.kps)

    def process(self, frame, cnt=0):
        lines = self.court_detector.detect(frame) if cnt == 0 else \
            self.court_detector.track_court(frame)

        ids, boxes, kps, kps_scores, ball, ball_score = self.HP.process(frame, print_time=True)
        # self.HP.visualize(frame)
        ids, boxes, kps, kps_scores = self.filter.filter(ids, boxes, kps, kps_scores, cnt)
        if self.write_json:
            self.Json.update(ids, boxes, kps, kps_scores, cnt)
        self.visualizer.visualize(frame, ids, boxes, kps, kps_scores, ball[0][0])
        self.court_detector.visualize(frame, lines)
        player_bv, move_bv, speed, hm = \
            self.top_view.process(self.court_detector, boxes[ids.tolist().index(1)],
                                  boxes[ids.tolist().index(2)])
        return frame, player_bv, move_bv, speed, hm

    def merge_img(self, frame, player_bv, move_bv, speed, hm):
        img_h, img_w = frame.shape[:2]
        player_bv = imutils.resize(player_bv, height=int(img_h/2))
        move_bv = imutils.resize(move_bv, height=int(img_h/2))
        speed = imutils.resize(speed, height=int(img_h/2))
        hm = imutils.resize(hm, height=int(img_h/2))
        hm = imutils.resize(hm, height=int(img_h/2))
        merged_img = np.concatenate((np.concatenate((player_bv, move_bv), axis=0),
                                     np.concatenate((speed, hm), axis=0)), axis=1)
        merged_img = imutils.resize(merged_img, height=img_h)
        return np.concatenate((frame, merged_img), axis=1)


    def release(self):
        if self.write_json:
            self.Json.release()


if __name__ == '__main__':
    from config import input_src
    import imutils
    import os
    cap = cv2.VideoCapture(input_src)
    out = cv2.VideoWriter("tmp/result.avi", cv2.VideoWriter_fourcc(*'MJPG'), 15, (2145, 800))
    os.makedirs("tmp", exist_ok=True)

    FP = FrameProcessor()

    idx = 0
    while True:
        ret, img = cap.read()
        if ret:
            imgs = FP.process(img, cnt=idx)
            img = FP.merge_img(*imgs)
            img = imutils.resize(img, height=800)
            cv2.imshow("result", img)
            cv2.waitKey(1)
            idx += 1
            out.write(img)
        else:
            out.release()
            break
