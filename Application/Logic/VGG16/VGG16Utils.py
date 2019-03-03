from Application.Logic.VGG16.VGG16Core import VGG16Core
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from Application.Logic.PCA.PCA import PCaUtils
from Application.Logic.LDA.LDA import LDaUtils
import os
TEST = True


class VGG16Utils:
    def __init__(self, version):
        self.prediction = []
        self.euclidean_dist = []
        self.first = False
        self.matched_id = ""
        self.score_id = 0
        self.n_dim = 0
        self.prob_distance = 0
        self.counter = 0
        self.path = version
        self.vgg = VGG16Core()
        self.id = self.load_id()
        self.gallery = self.load_gallery()
        self.gallery_size = self.gallery.shape[0]
        self.pca = PCaUtils(version)
        self.lda = LDaUtils(version)
        self.matched_from_video = False
        self.load_component4LDA = False
        self.accepted_probe = False
        try:
            dists = os.listdir(version+"Application/Logic/PCA/PCA_dataset/Distances")
            for dist in dists:
                os.remove(version+"Application/Logic/PCA/PCA_dataset/Distances/"+dist)
        except IOError as e:
            print("\n___No distances found___\n")

    def load_gallery(self):
        try:
            stored_gallery = np.matrix(np.loadtxt(self.path+'Dataset/Gallery/gallery.txt', dtype=np.float32))
            return stored_gallery
        except IOError:
            print('\n___No Gallery Found___\nMake sure that gallery exists\n')

    def load_id(self):
        try:
            identity = np.loadtxt(self.path+'Dataset/Gallery/id.txt',
                                  dtype=np.str)
            return identity
        except IOError:
            print("\n___No IDs Found___\nMake sure that gallery exists")

    def get_prediction(self, probe, need4gallery, threshold1, use_pca):
        self.prediction = self.vgg.make_prediction(img2be_processed=probe)
        if not need4gallery:
            sorted_distance = self.euclidean_distance(gallery=self.gallery[:, 1:],
                                                      prediction=self.prediction,
                                                      threshold1=threshold1,
                                                      metric='euclidean')
            self.lda.lda_projection(self.prediction)
            if TEST:
                print("\nMIN DISTANCE\t{:.06f}".format(sorted_distance[0]))
            if use_pca:
                pca_coord = self.pca.project_probe(self.prediction, n_dim=int(self.n_dim))
                sorted_pca_distance = self.euclidean_distance(gallery=pca_coord[:-1, :],
                                                              prediction=pca_coord[-1, :],
                                                              threshold1=0,
                                                              metric='cosine')
        return self.prediction

    def euclidean_distance(self, gallery, prediction, threshold1, metric):
        try:
            self.euclidean_dist = pairwise_distances(gallery,
                                                     np.matrix(prediction),
                                                     metric=metric)
            min_index = np.unravel_index(np.argmin(self.euclidean_dist),
                                         self.euclidean_dist.shape)[0]
            if min_index > self.gallery_size:
                self.matched_from_video = True
            else:
                self.matched_from_video = False
            sorted_array = sorted(np.squeeze(np.asarray(self.euclidean_dist)))
            self.score_id = sorted_array[1]/sorted_array[0]
            self.matched_id = self.id[min_index, 1]
            self.prob_distance = sorted_array[0]
            print("PD")
            if sorted_array[0] < np.float(threshold1):
                print("ERR")
                self.add2gallery_temp()
                self.accepted_probe = True
            else:
                self.accepted_probe = False
            return sorted_array
        except IOError:
            print("\n__Foundend just 1 id in gallery set__\n")
        except IndexError:
            print("")

    def add2gallery_temp(self):
        self.gallery = np.vstack([self.gallery,
                                  np.append(self.gallery.shape[0],
                                            self.prediction)])
        self.id = np.vstack([self.id,
                             np.append(self.gallery.shape[0],
                                       self.matched_id)])
