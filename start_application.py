from Application.Graphic.VideoCaptureGUI import VideoCaptureGUI
import numpy as np


class StartApplication:
    def __init__(self, lab):
        path = np.loadtxt("./Application/path.txt", dtype=np.str)
        if lab:
            version = "/home/spizzimenti/PycharmProjects/face_recognition/"
        else:
            version = str(path)
        VideoCaptureGUI(version=version).root.mainloop()


app = StartApplication(lab=False)
