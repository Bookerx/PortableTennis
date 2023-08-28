import numpy as np
import cv2
from scipy import signal
import imutils
import matplotlib.pyplot as plt
from .plot import plot_speed, calculate_point_frequencies


class TopViewProcessor:
    def __init__(self, lines):
        self.court = cv2.cvtColor(cv2.imread('src/court/court_reference.png'), cv2.COLOR_BGR2GRAY)
        # court = cv2.line(self.court, lines, 255, 5)
        # v_width, v_height = self.court.shape[::-1]
        self.court = cv2.cvtColor(self.court, cv2.COLOR_GRAY2BGR)
        # self.out = cv2.VideoWriter('minimap.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 12,
        #                       (v_width, v_height))
        self.inv_mats = []
        self.position1, self.position2 = [], []

    def process(self, court_detector, player1_box, player2_box):
        # img_h, img_w = inp_frame.shape[:2]
        """
        Calculate the feet position of both players using the inverse transformation of the court and the boxes
        of both players
        """
        inv_mats = court_detector.game_warp_matrix
        # self.inv_mats
        # positions_1 = []
        # positions_2 = []
        # Bottom player feet locations
        # for i, box in enumerate(player_1_boxes):
        feet_pos = np.array([(player1_box[0] + (player1_box[2] - player1_box[0]) / 2).item(), player1_box[3].item()]).reshape((1, 1, 2))
        feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[-1]).reshape(-1)
        self.position1.append(feet_court_pos)
        mask = []
        # Top player feet locations
        # for i, box in enumerate(player_2_boxes):
        feet_pos = np.array([(player2_box[0] + (player2_box[2] - player2_box[0]) / 2), player2_box[3]])\
            .reshape((1, 1, 2))
        feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[-1]).reshape(-1)
        self.position2.append(feet_court_pos)
        mask.append(True)

        # Smooth both feet locations
        window = 7 if len(self.position1) > 7 else len(self.position1)
        if window >= 7:
            positions_1 = np.array(self.position1)
            smoothed_1 = np.zeros_like(positions_1)
            smoothed_1[:,0] = signal.savgol_filter(positions_1[:,0], window, 2)
            smoothed_1[:,1] = signal.savgol_filter(positions_1[:,1], window, 2)
            positions_2 = np.array(self.position2)
            smoothed_2 = np.zeros_like(positions_2)
            smoothed_2[:,0] = signal.savgol_filter(positions_2[:,0], window, 2)
            smoothed_2[:,1] = signal.savgol_filter(positions_2[:,1], window, 2)

            # smoothed_2[not mask, :] = [None, None]
            frame = cv2.circle(self.court.copy(), (int(smoothed_1[-1][0]), int(smoothed_1[-1][1])), 45, (255, 0, 0), -1)
            frame = cv2.circle(frame, (int(smoothed_2[-1][0]), int(smoothed_2[-1][1])), 45, (255, 0, 0), -1)
        else:
            frame = self.court.copy()
        # width = frame.shape[1] // 7
        # resized = imutils.resize(frame, height=int(img_h/2))
        # cv2.imshow("mipmap", resized)
        movement = self.vis_movement()
        # movement = imutils.resize(movement, height=int(img_h/2))
        speed = self.vis_speed()
        # speed = imutils.resize(speed, height=int(img_h/2))
        hm = self.vis_heatmap()
        # hm = imutils.resize(hm, height=int(img_h/2))
        # merged_img = np.concatenate((np.concatenate((resized, movement), axis=0),
        #                              np.concatenate((speed, hm), axis=0)), axis=1)
        # merged_img = imutils.resize(merged_img, height=img_h)
        # cv2.imshow("merged_result", merged_img)
        return frame, movement, speed, hm
        # return np.concatenate((inp_frame, merged_img), axis=1)

        # return smoothed_1, smoothed_2

    def vis(self, player1_pos, player2_pos):
        plt.plot(player1_pos)

    def draw_movement_line(self, ls, t, frame):
        for i in range(t):
            cv2.line(frame, (int(ls[-i][0]), int(ls[-i][1])), (int(ls[-i - 1][0]), int(ls[-i - 1][1])),
                     (0, 255, 0), 5)

    def vis_movement(self):
        max_movement_time = 5
        frame = self.court.copy()
        lines_num1 = min(max_movement_time, len(self.position1)-1)
        lines_num2 = min(max_movement_time, len(self.position2)-1)
        self.draw_movement_line(self.position1, lines_num1, frame)
        self.draw_movement_line(self.position2, lines_num2, frame)
        return frame
        # width = frame.shape[1] // 7
        # resized = imutils.resize(frame, width=width)
        # cv2.imshow("movement", resized)

    def vis_speed(self):
        max_speed_time = 20
        plot_speed(self.position1, self.position2, max_speed_time)
        img = cv2.imread("tmp/speed_tmp.png")
        return img
        cv2.imshow("speed", img)

    def vis_heatmap(self):
        plt.clf()
        concat_pos = np.concatenate((self.position1, self.position2), axis=0)
        fre_map = calculate_point_frequencies(self.court.shape[1], self.court.shape[0], concat_pos, 6, 6)
        plt.imshow(np.array(fre_map))
        plt.savefig("tmp/heatmap_tmp.png")
        plt.clf()
        return cv2.imread("tmp/heatmap_tmp.png")
        cv2.imshow("heatmap", cv2.imread("heatmap_tmp.png"))



