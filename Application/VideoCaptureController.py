from Application.Logic.VideoCaptureLogic import VideoCaptureLogic
import os
import cv2

TEST = True


class VideoCaptureController:
    def __init__(self, version):
        self.Logic = VideoCaptureLogic(version=version)
        self.path = version
        self.video_pointer = 0

    def use_pca(self, value):
        if value == 1:
            self.Logic.use_pca = True
        else:
            self.Logic.use_pca = False

    def ask4load_components(self, value):
        if value == 1:
            self.Logic.vgg.pca.load_component = True
        else:
            self.Logic.vgg.pca.load_component = False

    def set_threshold4temp_template(self, val):
        self.Logic.threshold4temp_template = val

    def change_pca_dimensions(self, val):
        self.Logic.vgg.n_dim = val
        self.Logic.vgg.pca.projection(self.Logic.vgg.pca.gallery, int(val))
        self.Logic.vgg.pca.load()
        self.Logic.vgg.counter += 1

# The folder ./Media contains videos to be analyzed
    def next_video(self):
        self.stack = os.listdir(self.path+'Media')
        self.Logic.video = cv2.VideoCapture(self.path+'Media/'+self.stack[self.video_pointer])
        self.video_pointer += 1

    def use_prediction(self, value):
        if value == 1:
            self.Logic.pred = True
        else:
            self.Logic.pred = False

