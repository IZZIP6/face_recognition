from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
from time import time
import os

TEST = True


class PCaUtils:
    def __init__(self, version):
        self.sc = StandardScaler()
        self.path = version
        self.gallery = np.matrix
        self.load_gallery()
        self.load_component = False
        self.load()
        self.coord_gallery = np.matrix
        try:
            os.remove(self.path + 'Application/Logic/PCA/PCA_dataset/coordinates.txt')
        except OSError:
            print("\n___Nothing to del___\n")
        try:
            self.coord_file = open(self.path + 'Application/Logic/PCA/PCA_dataset/coordinates.txt', 'a')
        except IOError:
            print("\n___No such file___\n")

    def load(self):
        try:
            self.space_components = np.loadtxt(self.path+'Application/Logic/PCA/PCA_dataset/components_by_proj.txt',
                                               dtype=np.float32)
            self.coord_gallery = np.matmul(self.gallery[:, :], self.space_components.transpose())
        except IOError:
            print("\n___No space component Found. Compute them using PCA___\n")

    def load_gallery(self):
        try:
            loaded_gallery = np.loadtxt(self.path+'Dataset/Gallery/gallery.txt',
                                        dtype=np.float32)
            self.gallery = np.matrix(loaded_gallery)[:, 1:]
        except IOError:
            print("No found gallery")

    def prepare_data(self, gallery):
        preprocessed_gallery = gallery
        preprocessed_gallery = self.sc.fit_transform(preprocessed_gallery)
        return preprocessed_gallery

    def projection(self, gallery, components):
        gallery4projection = self.prepare_data(gallery)
        sklearn_pca = PCA(n_components=components)
        new_coords = sklearn_pca.fit_transform(gallery4projection)
        component4new_space = sklearn_pca.components_
        np.savetxt(self.coord_file, np.c_[new_coords], fmt="%s", delimiter="\t")
        return new_coords

    def project_probe(self, probe, n_dim):
        if self.load_component:
            coords = self.space_components.transpose()
            new_coords = np.matmul(probe, self.space_components.transpose())
            np.savetxt(self.coord_file,
                       np.c_[new_coords],
                       fmt="%s",
                       delimiter="\t")
            return np.vstack([self.coord_gallery, new_coords])
        else:
            return self.projection(np.vstack([self.gallery, probe]), n_dim)
