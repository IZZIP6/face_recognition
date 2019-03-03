from Application.Graphic.VideoCaptureGUI import VideoCaptureGUI


class StartApplication:
    def __init__(self, lab):
        if lab:
            version = "/home/spizzimenti/PycharmProjects/face_recognition/"
        else:
            version = "C:/Users/pizzi/Desktop/face_recognition/"     #set directory path
        VideoCaptureGUI(version=version).root.mainloop()


app = StartApplication(lab=False)
