from classes.image_viewer import ImageViewer
import numpy as np
import cv2

class Kmeans():
    def __init__(self, input_viewer, output_viewer,max_num_of_iterations = 100 ):
        self.input_viewer = input_viewer
        self.output_viewer = output_viewer
        self.num_of_iterations = max_num_of_iterations
        self.convergence_thereshold = 1e-3
    def apply_kmeans(self, k):
        print("apply the k mean clustering")
        self.num_of_clusters = k
        self.clusters = [[] for _ in range(self.num_of_clusters)] # empty lists based on the number of clusters to contain the same cluster
        self.feature_space = self.get_feature_space()
        self.centroids = self.initalize_centroids()
        self.make_clustering()
        self.apply_clustering_on_image()

    def get_feature_space(self):
        # take the feature space of self.input_viewer.current_image with the location
        image_bgr = self.input_viewer.current_image.modified_image
        image_rgb = cv2.cvtColor(self.input_viewer.current_image.modified_image, cv2.COLOR_BGR2RGB)
        # image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h,w,c = image_rgb.shape

        # extract the coordinate to encourge the closer to be the same cluster
        x_coords, y_coords = np.meshgrid(np.arange(w),np.arange(h)) # get locations as we have w , h  only

        # normalize the x and y coords
        x_coords = x_coords/w
        y_coords = y_coords/h

        # get the vector for each feature (5d space)
        # hs_features = image_hsv[:, :, :2].reshape(-1, 2).astype(np.float32)
        rgb = image_rgb.reshape(-1, 3).astype(np.float32) # for each pixel i we have vector [r g b]
        x_coords = x_coords.reshape(-1, 1) # to get the x coord for each pixel "flatten"
        y_coords = y_coords.reshape(-1, 1) # to get y coord for each pixel

        feature_space = np.hstack((rgb, x_coords, y_coords)) # for each pixel we have [r g b x y]
        return feature_space

    def initalize_centroids(self):
        '''initalization may be random or uniform or apply on small subset then start with it's means .
        kmeans is very sensitive to outliers and if not initialize well may be not converge
        subset ensure faster convergence'''

        # start with random select 1000 points to be subset
        num_samples = min(1000, len(self.feature_space)) # to avoid take much than length of the feature space
        subset_indices = np.random.choice(len(self.feature_space), size=num_samples, replace=False)
        subset = self.feature_space[subset_indices]

        # random select k points to represent the centroid
        initial_indices = np.random.choice(num_samples, size=self.num_of_clusters, replace=False)
        initial_centroids = subset[initial_indices]

        # make clustering on this subset
        centroids, _ = self.make_clustering(feature_space=subset, centroids=initial_centroids, num_iterations=10)
        return centroids

    def make_clustering(self, feature_space=None, centroids=None, num_iterations=None):
        # if not provided, use defaults (full dataset, current centroids)
        if feature_space is None:
            feature_space = self.feature_space
        if centroids is None:
            centroids = self.centroids
        if num_iterations is None:
            num_iterations = self.num_of_iterations
        # to stop after reahcing max iterations
        for _ in range(num_iterations):
            clusters = [[] for _ in range(self.num_of_clusters)]
            for sample in feature_space: # loop over sample in feature space
                distances = self.euclidean_distance(sample, centroids)
                idx = self.get_idx_cloest_centroid(distances)  # so i know idx of cluster
                clusters[idx].append(sample)

            new_centroids = []
            for cluster in clusters:
                if cluster: # calc mean of this cluster to be the new centroids
                    new_centroids.append(np.mean(cluster, axis=0))
                else:
                    # for empty cluster pick ranfom point
                    random_idx = np.random.randint(len(feature_space))
                    new_centroids.append(feature_space[random_idx])
            new_centroids = np.array(new_centroids)
             # show convergence
            diff = np.linalg.norm(centroids - new_centroids)
            if diff < self.convergence_thereshold:
                print(f"the iteration num is {_}")
                break

            self.centroids = new_centroids
            self.clusters =clusters


        return self.centroids, self.clusters
    def get_idx_cloest_centroid(self, distances): # to assign this sample data to closet centroid in order to put it in cluster
        return np.argmin(distances)
    def euclidean_distance(self,sample ,centroids):
        return np.linalg.norm(centroids - sample, axis=1)

    def apply_clustering_on_image(self):
        print("apply colors based on centroids")

        img = self.input_viewer.current_image.modified_image
        h, w, c = img.shape

        output_img = np.zeros((h, w, c), dtype=np.uint8)


        idx = 0
        for y in range(h):
            for x in range(w):
                sample = self.feature_space[idx]  # Get the feature vector (RGB + coords)
                distances = self.euclidean_distance(sample, self.centroids)  # Get distances to centroids
                cluster_idx = self.get_idx_cloest_centroid(distances)  # Find closest centroid

                # Assign the color of the closest centroid to the pixel
                output_img[y, x] = self.centroids[cluster_idx][:3]  # Extract RGB from the centroid (5D -> 3D)

                idx += 1


        self.output_viewer.current_image.modified_image = output_img



