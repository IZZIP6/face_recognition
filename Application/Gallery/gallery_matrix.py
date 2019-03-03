from Application.Logic.VGG16.VGG16Utils import VGG16Utils
import numpy as np

class watch_list:

    def __init__(self, version):
        self.vgg = VGG16Utils(version)
        self.path = version
        self.number_id = 0
        self.id = {}
        self.load_gallery()

    def load_gallery(self):
        try:
            self.loaded_gallery = np.loadtxt(self.path+'Dataset/Gallery/gallery.txt', dtype=np.float32)
            self.loaded_id = np.loadtxt(self.path+'Dataset/Gallery/id.txt', dtype=np.str)
            self.gallery_w_label = np.matrix(self.loaded_gallery)
            self.number_id = self.gallery_w_label.shape[0]
            print(self.number_id)
        except IOError:
            print('No gallery found')
        return self.gallery_w_label

    def add_id(self, img, name):
        self.id_name = name
        prediction = self.vgg.get_prediction(img, need4gallery=True, threshold1=None, threshold2=None, use_pca=False)
        prediction_w_label = np.append(self.number_id, prediction)
        self.gallery = np.matrix(prediction)
        print(self.gallery_w_label.shape, "\t", prediction_w_label.shape)
        self.gallery_w_label = np.vstack([self.gallery_w_label, prediction_w_label])
        self.store_gallery()
        self.fill_dict()

    def store_gallery(self):
        np.savetxt('C:/Users/pizzi/Desktop/face_recognition/Dataset/Gallery/gallery.txt', np.c_[self.gallery_w_label], fmt="%.8f", delimiter="\t") #change path

    def fill_dict(self):
        self.loaded_id = np.vstack([self.loaded_id, [self.number_id, self.id_name]])
        np.savetxt('C:/Users/pizzi/Desktop/face_recognition/Dataset/Gallery/for_slide/id.txt', np.c_[self.loaded_id], fmt="%s", delimiter="\t") #change path
        print(self.loaded_id)
