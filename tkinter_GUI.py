import tkinter as tk
from tkinter import ttk
from tkinter import font
import cv2
from PIL import Image, ImageTk
import os
import config

LARGE_FONT = ("Verdana", 12)
video_src = config.input_src
geometric = (1280, 720)
from main import FrameProcessor


def update_image(canvas, image, size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor="nw", image=photo)
    canvas.image = photo


def load_image(canvas, path, size):
    image = cv2.imread(path)
    update_image(canvas, image, size)


class Demo(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("{0}x{1}".format(self.winfo_screenwidth(), self.winfo_screenheight()))
        # self.geometry("{0}x{1}".format(geometric[0], geometric[1]))

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, FontCheck):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.calib_flag = False
        self.start_flag = False

        self.label = tk.Label(self, text="Portable Tennis System", font=LARGE_FONT)
        self.label.pack(pady=10, padx=10)
        win_width, win_height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.win_width, self.win_height = win_width, win_height
        # video to be played
        self.canvas_video = tk.Canvas(self, width=win_width*0.55, height=win_height*0.55, bg="black")
        self.canvas_video.pack()
        tk.Label(self, text="Video", font=("Verdana", 20)).place(relx=0.475, rely=0.05, anchor="center")
        # speed map
        self.canvas_speed = tk.Canvas(self, width=win_width*0.19, height=win_width*0.19*0.75, bg="black")
        self.canvas_speed.pack()
        self.speed_size = (int(win_width * 0.2), int(win_width * 0.2*0.75))
        tk.Label(self, text="Speed Map", font=("Verdana", 14)).place(relx=0.875, rely=0.055, anchor="center")

        # heat map
        self.canvas_heatmap = tk.Canvas(self, width=win_width*0.19, height=win_width*0.19*0.75, bg="black")
        self.canvas_heatmap.pack()
        tk.Label(self, text="Heatmap", font=("Verdana", 14)).place(relx=0.875, rely=0.3525, anchor="center")

        # birdseyeview of field (players)
        self.canvas_players = tk.Canvas(self, width=win_width*0.1, height=win_width*0.1*64/36, bg="black")
        self.canvas_players.pack()
        tk.Label(self, text="Player", font=("Verdana", 14)).place(relx=0.825, rely=0.6425, anchor="center")

        # birdseyeview of field (ball)
        self.canvas_ball = tk.Canvas(self, width=win_width*0.1, height=win_width*0.1*64/36, bg="black")
        self.canvas_ball.pack()
        tk.Label(self, text="Ball", font=("Verdana", 14)).place(relx=0.935, rely=0.6425, anchor="center")

        # player 1 skeleton
        self.canvas_player1 = tk.Canvas(self, width=win_height*0.25, height=win_height*0.25, bg="black")
        self.canvas_player1.pack()
        tk.Label(self, text="Player 1 skeleton", font=("Verdana", 14)).place(relx=0.0925, rely=0.055, anchor="center")

        # player 2 skeleton
        self.canvas_player2 = tk.Canvas(self, width=win_height*0.25, height=win_height*0.25, bg="black")
        self.canvas_player2.pack()
        tk.Label(self, text="Player 2 skeleton", font=("Verdana", 14)).place(relx=0.0925, rely=0.365, anchor="center")

        self.button_calibrate = tk.Button(self, text="Calibrate", font=("Verdana", 30),
                                     command=self.begin_calibration)
        self.button_calibrate.pack()

        self.button_start = tk.Button(self, text="Start", font=("Verdana", 30), command=self.begin_process)
        self.button_start.pack()

        self.button_report = tk.Button(self, text="Report", font=("Verdana", 30), command=lambda: controller.show_frame(PageOne))
        self.button_report.pack()

        self.gui_set()
        self.canvas_sizes()

    def canvas_sizes(self):
        self.video_size = (int(self.win_width*0.55), int(self.win_height*0.55))
        self.speed_size = (int(self.win_width * 0.2), int(self.win_width * 0.2*0.75))
        self.heatmap_size = (int(self.win_width * 0.2), int(self.win_width * 0.2*0.75))
        self.mipmap_size = (int(self.win_width * 0.1), int(self.win_width * 0.1*64/36))
        self.player_move_size = (int(self.win_height * 0.25), int(self.win_height * 0.25))

    def gui_set(self):
        self.canvas_video.place(relx=0.475, rely=0.35, anchor="center")
        self.canvas_ball.place(relx=0.935, rely=(0.66+0.05*256/81), anchor="center")
        self.canvas_speed.place(relx=0.875, rely=0.2, anchor="center")
        self.canvas_player1.place(relx=0.0925, rely=0.195, anchor="center")
        self.canvas_players.place(relx=0.825, rely=0.66+0.05*256/81, anchor="center")
        self.canvas_player2.place(relx=0.0925, rely=0.505, anchor="center")
        self.canvas_heatmap.place(relx=0.875, rely=0.5, anchor="center")

        self.button_calibrate.place(relx=0.125, rely=0.825, anchor="center", relwidth=0.2, relheight=0.2)
        self.button_start.place(relx=0.375, rely=0.825, anchor="center", relwidth=0.2, relheight=0.2)
        self.button_report.place(relx=0.625, rely=0.825, anchor="center", relwidth=0.2, relheight=0.2)

        self.button_start.config(state="disabled")
        self.button_report.config(state="disabled")

    def init_video(self):
        self.cap = cv2.VideoCapture(video_src)
        os.makedirs("tmp", exist_ok=True)
        self.FP = FrameProcessor()
        self.idx = 0
        ret, img = self.cap.read()
        img = self.FP.process(img, cnt=self.idx)[0]
        return img

    def begin_calibration(self):
        if self.calib_flag:
            self.calib_flag = False
        else:
            self.calib_flag = True
            self.button_calibrate.config(text="Calibrating...")
            init_img = self.init_video()
            update_image(self.canvas_video, init_img, size=self.video_size)
            self.button_calibrate.config(text="Calibrating...", state="disabled")
            self.button_start.config(state="normal")

    def begin_process(self):
        if self.start_flag:
            self.start_flag = False
            self.button_report.config(state="normal")
            self.button_start.config(text="Start")
        else:
            self.start_flag = True
            self.button_start.config(text="Stop")
            while True:
                if self.start_flag:
                    ret, img = self.cap.read()
                    if not ret:
                        break
                    img, player_bv, move_bv, speed, hm = self.FP.process(img, cnt=self.idx)
                    update_image(self.canvas_video, img, size=self.video_size)
                    self.idx += 1
                    update_image(self.canvas_speed, speed, size=self.speed_size)
                    update_image(self.canvas_heatmap, hm, size=self.heatmap_size)
                    # update_image(self.canvas_player1, player_bv[0], size=self.player_move_size)
                    # update_image(self.canvas_player2, player_bv[1], size=self.player_move_size)
                    update_image(self.canvas_ball, move_bv, size=self.mipmap_size)
                    update_image(self.canvas_players, player_bv, size=self.player_move_size)
                else:
                    self.start_flag = False
                    break




class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        win_width, win_height = self.winfo_screenwidth(), self.winfo_screenheight()

        canvas1 = tk.Canvas(self, width=win_width * 0.3, height=win_width * 0.75 * 0.3, bg="white")
        canvas1.pack()
        canvas1.place(relx=0.225, rely=0.7, anchor="center")

        canvas_graph1 = tk.Canvas(self, width=win_width*0.3, height=win_width*0.75*0.3, bg="black")
        canvas_graph1.pack()
        canvas_graph1.place(relx=0.225, rely=0.275, anchor="center")

        canvas_heatmap = tk.Canvas(self, width=win_width*0.3, height=win_height*0.3, bg="black")
        canvas_heatmap.pack()
        canvas_heatmap.place(relx=0.8, rely=0.2, anchor="center")

        canvas_players = tk.Canvas(self, width=win_width*0.15, height=win_width*0.15*16/9, bg="black")
        canvas_players.pack()
        canvas_players.place(relx=0.715, rely=0.4, anchor="nw")

        button_back = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button_back.pack()
        button_back.place(relx=0.02, rely=0.02, anchor='nw')

        image_button = tk.Button(self, text="Load Image", command=lambda: load_image(canvas1, "image/sample/IMG_7330.JPG",
                                                                                     size=(400, 300)))
        image_button.pack()
        image_button.place(relx=0.1, rely=0.95, anchor='center')

        button_report = tk.Button(self, text="Tmp")
        button_report.place(relx=0.2, rely=0.95, anchor='center')

        selectionVar = tk.StringVar()
        select_combobox = ttk.Combobox(self, textvariable=selectionVar)
        select_combobox['values'] = ('just', 'random', 'entries')
        select_combobox['state'] = 'readonly'
        select_combobox.place(relx=0.5, rely=0.05, anchor='center')

class FontCheck(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        win_width, win_height = self.winfo_screenwidth(), self.winfo_screenheight()
        # win_width, win_height = geometric[0], geometric[1]

        fonts = list(font.families())
        fonts.sort()

        def populate(frame):
            # Put in the fonts
            listnumber = 1
            for item in fonts:
                label = "listlabel" + str(listnumber)
                label = tk.Label(frame, text=item, font=(item, 16)).pack()
                listnumber += 1

        def onFrameConfigure(canvas):
            # Reset the scroll region to encompass the inner frame
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")
        frame = tk.Frame(canvas, background="#ffffff")
        vsb = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((4, 4), window=frame, anchor="nw")

        frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))

        populate(frame)

app = Demo()
app.mainloop()
