import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

PERCENTAGE = 0.8
TEST = True


class LDaUtils:
    def __init__(self, path):
        self.path = path
        self.gallery = np.matrix
        self.known = self.gallery
        self.unknown = self.gallery
        self.target4know = np.zeros
        self.target4unknow = np.zeros
        self.model = LinearDiscriminantAnalysis(store_covariance=True)
        self.unknown = False
        self.load_gallery()

    def load_gallery(self):
        try:
            self.gallery = np.matrix(np.loadtxt(self.path+'\Dataset\Gallery\gallery.txt',
                                                dtype=np.float32))[:, 1:]
        except IOError:
            print("\n___No such file___\nMake sure that the gallery exists")
        self.gallery = self.prepare_data(self.gallery)
        index = int(round(self.gallery.shape[0] * PERCENTAGE))
        self.known = self.gallery[:index, :]
        self.unknown = self.gallery[index:-1, :]
        self.target4know = np.zeros(self.known.shape[0])
        self.target4unknow = np.zeros(self.unknown.shape[0]) + 1
        self.model_lda(self.gallery[:-1, :],
                       np.hstack([self.target4know, self.target4unknow]))

    @staticmethod
    def prepare_data(data):
        sc = StandardScaler()
        return sc.fit_transform(data)

    def model_lda(self, data, data_label):
        coords = self.model.fit_transform(data, data_label)
        components = self.model.scalings_
        np.savetxt(self.path + 'Application/Logic/LDA/LDA_dataset/space_components.txt',
                   np.c_[components],
                   fmt="%s",
                   delimiter="\t")
        np.savetxt(self.path+'Application/Logic/LDA/LDA_dataset/know_unknow_coords.txt',
                   np.c_[coords],
                   fmt="%s",
                   delimiter="\t")

    def lda_projection(self, data):
        # data = self.prepare_data(data)
        if TEST:
            print(self.model.tol)
        predict = self.model.predict(np.matrix(data))
        coords = np.matmul(self.model.scalings_.transpose(), np.matrix(data).transpose())
        if abs(coords) > 0.08:
            print("\nSCONOSCIUTO")
            self.unknown = True
        else:
            self.unknown = False
        if TEST:
            print("\n.:.:.:.:.:.COORDINATE.:.:.:.:.:.", coords)
