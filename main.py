from src.human_detection import HumanDetector
import cv2
import config.config as config
from utils.generate_json import JsonGenerator
from utils.filter_result import ResultFilterer
from utils.visualize import Visualizer
from src.court.court_detector import CourtDetector
from src.court.top_view import TopViewProcessor

detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg = config.detector_cfg, \
                                config.detector_weight, config.pose_weight, config.pose_model_cfg, config.pose_data_cfg
write_json = config.write_json
filter_criterion = config.filter_criterion


class FrameProcessor:
    def __init__(self):
        self.HP = HumanDetector(detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg)
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

        ids, boxes, kps, kps_scores = self.HP.process(frame, print_time=True)
        # self.HP.visualize(frame)
        ids, boxes, kps, kps_scores = self.filter.filter(ids, boxes, kps, kps_scores, cnt)
        if self.write_json:
            self.Json.update(ids, boxes, kps, kps_scores, cnt)
        self.visualizer.visualize(frame, ids, boxes, kps, kps_scores)
        self.court_detector.visualize(frame, lines)
        frame = self.top_view.process(self.court_detector, boxes[0], boxes[1], frame)
        return frame

    def release(self):
        if self.write_json:
            self.Json.release()


if __name__ == '__main__':
    from config.config import input_src
    import imutils
    cap = cv2.VideoCapture(input_src)

    FP = FrameProcessor()

    idx = 0
    while True:
        ret, img = cap.read()
        if ret:
            img = FP.process(img, cnt=idx)
            img = imutils.resize(img, height=800)
            cv2.imshow("result", img)
            cv2.waitKey(1)
            idx += 1
        else:
            break
