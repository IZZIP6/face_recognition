from Application.Logic.VGG16.VGG16Utils import VGG16Utils
from PIL import Image, ImageTk
import face_recognition
import cv2


class VideoCaptureLogic:
    def __init__(self, version):
        self.vgg = VGG16Utils(version=version)
        self.video = cv2.VideoCapture(0)
        # cv2.waitKey(1)
        self.face_locations = []
        self.top = []
        self.right = []
        self.bottom = []
        self.left = []
        self.threshold4temp_template = 0
        self.use_pca = False
        self.ret = False
        self.frame = []
        self.rgb_color = [255, 200, 50]
        self.pred = True

    def run_video(self):
        self.ret, self.frame = self.video.read()
        if self.ret:
            rgb_frame = self.frame[:, :, ::-1]
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            self.face_locations = face_recognition.face_locations(small_frame,
                                                                  number_of_times_to_upsample=2)
            for face in self.face_locations:
                self.top, self.right, self.bottom, self.left = face
                if self.pred:
                    self.prediction(self.top, self.right, self.left, self.bottom)
                cv2.rectangle(self.frame,
                              (self.left*4-10, self.top*4-10),
                              (self.right*4+10, self.bottom*4+10),
                              self.rgb_color,
                              2)

            image = cv2.cvtColor(self.frame,
                                 cv2.COLOR_BGR2RGBA)
            current_image = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=current_image)
            return imgtk

    def prediction(self, top, right, left, bottom):
        face = self.frame[top*4:bottom*4, left*4:right*4]
        self.vgg.get_prediction(cv2.resize(face, (224, 224)),
                                need4gallery=False,
                                threshold1=self.threshold4temp_template,
                                use_pca=self.use_pca)
        try:
            if not self.vgg.matched_from_video:
                self.rgb_color = [255, 200, 50]
            else:
                self.rgb_color = [185, 100, 10]
            cv2.rectangle(self.frame,
                          (left * 4 - 10, top * 4 - 40),
                          (right * 4 + 10, top * 4 - 8),
                          self.rgb_color,
                          cv2.FILLED)
            cv2.putText(self.frame,
                        self.vgg.matched_id,
                        (left*4-10, top*4-18),
                        cv2.FONT_ITALIC,
                        1,
                        color=(255, 255, 255))

        except AttributeError:
            print("\n___No one Found___\n")

    def show(self):
        print(self.threshold4temp_template)