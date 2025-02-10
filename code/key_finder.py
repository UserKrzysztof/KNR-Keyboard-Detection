import numpy as np
import pandas as pd

from skimage import io, color, filters, measure, morphology
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN, OPTICS
from scipy import spatial
import statsmodels.api as sm

import json

class KeyFinder:
    def __init__(self, 
                 min_keys=60, 
                 probability_threshold=0.8,
                 max_cluster_size = 100,
                 min_key_size=100,
                 key_displacement_distance=0.05,
                 use_gauss = False,
                 input_missing_keys=False,
                 check_space_eccentricity= True,
                 min_eccentricity = 0.25,
                 cluster_epsilon = 50,
                 keys_pattern_file='clean_letters.json'):
        assert keys_pattern_file is not None

        self.min_keys = min_keys
        self.probability_threshold = probability_threshold
        self.min_key_size = min_key_size
        self.key_displacement_distance = key_displacement_distance
        self.input_missing_keys = input_missing_keys
        self.use_gauss = use_gauss
        self.max_cluster_size = max_cluster_size
        self.check_space_eccentricity = check_space_eccentricity
        self.min_eccentricity = min_eccentricity
        self.cluster_epsilon = cluster_epsilon

        self.centers = []
        self.base1 = []
        self.base2 = []
        self.centers_base2 = []
        self.return_values = []
        self.space_coords = []
        self.found_map = {}
        self.not_swap = True

        with open(keys_pattern_file, 'r') as f:
            self.base_map = json.load(f)
            self.base_kdtree = spatial.KDTree(np.array(list(self.base_map.values())))

    def find(self, mask_for_keys):
        self.mask_to_centers(mask_for_keys)
        self.clear_centers()
        self.find_space()
        self.decompose()
        self.centers_to_map()
        return self.assign_keys()

    def mask_to_centers(self, mask_for_keys):
        label = measure.label(mask_for_keys.astype(bool))
        label = morphology.remove_small_objects(label, min_size=self.min_key_size)
        #label = morphology.remove_objects_by_distance(label, min_distance = self.cluster_epsilon)
        label = measure.label(label)
        regions = measure.regionprops(label)

        centroids = []
        self._props = []
        for i, props in enumerate(regions):
            y0, x0 = props.centroid
            centroids.append(np.array((x0,y0)))
            self._props.append(props)

        self.centers = np.array(centroids)
        print(self.centers.shape)
        return self.centers
    
    def find_space(self):
        assert len(self.centers) > 0
        space_area = 0
        space_coords = None
        for i, props in enumerate(self._props):
            is_valid = True
            if self.check_space_eccentricity:
                is_valid = props.eccentricity > self.min_eccentricity

            if props.area > space_area and is_valid:
                space_area = props.area
                space_coords = i

        self.space_coords = space_coords
        return self.centers


    def clear_centers(self):
        assert len(self.centers) > 0

        print("ELO")
        print(self.use_gauss)
        if self.use_gauss == True:
            print("ELO2")
            gmm = GaussianMixture(n_components=1)
            gmm.fit(self.centers)
            probabilities = gmm.predict_proba(self.centers)[:, 0]
            filtered_centers = self.centers[probabilities >= self.probability_threshold]
            self.centers = filtered_centers
        else:
            #clusterer = HDBSCAN(max_cluster_size=self.max_cluster_size, metric="cityblock")
            #clusterer = OPTICS(metric="cityblock", max_eps=self.cluster_epsilon)
            #cluster_labels = clusterer.fit_predict(self.centers)
            #self.centers = self.centers[cluster_labels != -1]
            #indices = np.where(cluster_labels != -1)[0]
            #self._props = [self._props[i] for i in indices]
            X, Y = self.centers[:, 0].reshape(-1, 1), self.centers[:, 1]
            print(X,Y)

            def compute_outliers(X, Y):
                model = sm.OLS(Y, sm.add_constant(X)).fit()
                influence = model.get_influence()
                student_resid = influence.resid_studentized_external
                leverage = influence.hat_matrix_diag
                
                outlier_mask = np.abs(student_resid) > 2  
                influence_mask = leverage > (2 * np.mean(leverage))  
                
                return ~(outlier_mask | influence_mask)  
            
            mask_x = compute_outliers(X, Y)
            mask_y = compute_outliers(Y.reshape(-1, 1), X.flatten())
            final_mask = mask_x & mask_y

            self.centers = self.centers[final_mask]
            indices = np.where(final_mask)[0]
            self._props = [self._props[i] for i in indices]

        return self.centers

    def decompose(self):
        assert len(self.centers) > self.min_keys, "Not enough centers were found"
        
        centroids = np.array(self.centers) - np.mean(self.centers, axis=0)
        _, _, VT = np.linalg.svd(centroids)
        
        base2 = VT 
        self.base1 = [(1,0),(0,1)]
        self.base2 = [np.array(vec) for vec in base2]

        if np.matmul(self.base2[1], centroids[self.space_coords,:]) > 0:
            self.base2[1] = -1*self.base2[1]
        if np.matmul(self.base2[0], centroids[self.space_coords,:]) > 0:
            self.base2[0] = -1*self.base2[0]
        return self.base2

    def centers_to_map(self):
        M_base1_base2 = np.matrix(self.base2)
        centroids = np.matrix(np.array(self.centers) - np.mean(self.centers, axis=0))
        self.centers_base2 = np.matmul(centroids, M_base1_base2)
        self.return_values = [
            np.min(self.centers_base2[:,0]),
            np.max(self.centers_base2[:,0]),
            np.min(self.centers_base2[:,1]),
            np.max(self.centers_base2[:,1]),
        ]
        return self.centers_base2

    def assign_keys(self):
        assert len(self.centers_base2) > 0
        coords = self.centers_base2.copy()
        print(coords.shape)
        coords[:,0] = (coords[:,0] - np.min(coords[:,0]))/(np.max(coords[:,0]) - np.min(coords[:,0]))
        coords[:,1] = (coords[:,1] - np.min(coords[:,1]))/(np.max(coords[:,1]) - np.min(coords[:,1]))

        keys = list(self.base_map.keys())
        idx_to_pop = []
        for i in range(coords.shape[0]):
            distance, idx = self.base_kdtree.query(coords[i,:])
            distance = distance[0]
            idx = idx[0]
            key = keys[idx]
            idx_to_pop.append(idx)

            value = self.found_map.get(key, None)
            if value is None:
                self.found_map[key] = {
                    "coords":coords[i,:], 
                    "dist": distance
                }
            else:
                old_dist = value["dist"]
                if abs(distance - old_dist) < self.key_displacement_distance:
                    self.found_map[key] = {
                    "coords":coords[i,:], 
                    "dist": distance
                }

        if self.input_missing_keys:
            keys = np.delete(keys, idx_to_pop)      
            for key in keys:
                coords = self.base_map[key]
                self.found_map[key] = {
                        "coords":np.matrix(coords), 
                        "dist": .0
                    }
                
        return self.found_map
    
        
