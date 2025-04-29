from classes.image_viewer import ImageViewer
import numpy as np
import cv2
from classes.Kmeans import Kmeans
from skimage.segmentation import slic
from skimage.measure import regionprops
class Agglomerative():
    def __init__(self, input_viewer, output_viewer,max_num_of_iterations = 10 ):
        self.input_viewer = input_viewer
        self.output_viewer = output_viewer
        self.num_of_iterations = max_num_of_iterations
        self.superpixel_centroids = None
        self.pixel_assignments = None


    def apply_agglomerative(self, k):
        print("apply agglomerative clustering ")
        self.num_of_clusters = k
        self.feature_space = self.get_feature_space()
        self.make_agglomerative()
        self.apply_clustering_on_image()
    def get_feature_space(self):
        print("iam in get feature")
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
        rgb = image_rgb.reshape(-1, 3).astype(np.float32)/255.0 # for each pixel i we have vector [r g b]
        x_coords = x_coords.reshape(-1, 1) # to get the x coord for each pixel "flatten"
        y_coords = y_coords.reshape(-1, 1) # to get y coord for each pixel
        full_feature_space = np.hstack((rgb, x_coords, y_coords)) # for each pixel we have [r g b x y]
        print("getting super pixel")
        # convert image to 5000 super pixel instead as it will be time and space computionally expensive to make it on the orignla pixels

        num_initial_clusters = 1000
        labels = slic(image_rgb, n_segments=num_initial_clusters, compactness=10, sigma=1)
        regions = regionprops(labels, intensity_image=image_rgb)
        kmeans_centroids = np.array(
            [[r.mean_intensity[0] / 255.0, r.mean_intensity[1] / 255.0, r.mean_intensity[2] / 255.0,
              r.centroid[1] / w , r.centroid[0] / h ] for r in regions])
        self.superpixel_centroids = kmeans_centroids
        self.pixel_assignments = labels.flatten()
        return kmeans_centroids

    def make_agglomerative(self):
        print("make the agglomerative")
        # first we assume each one of our super pixel as cluster
        feature_space = self.feature_space
        n_samples = feature_space.shape[0] # get number of super pixels
        clusters = {i: [i] for i in range(n_samples)}  # Make it a dictionary
        # second : compute distance between clusters in distance matrix
        distances= {}
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                points_c1 = feature_space[clusters[i]]
                points_c2 = feature_space[clusters[j]]
                dist = self.ward_distance(points_c1, points_c2)
                distances[(i, j)] = dist
                '''
                ward's linkage is merging method trys to minmize the increase of variance after merging 
                so it's not about the current , it choose the min increase after meriging 
                '''
        current_clusters = list(clusters.keys())
        iteration = 0
        while len(current_clusters) > self.num_of_clusters :
            iteration += 1
            # third : we found the most two similar pair to merge
            min_pair = None
            min_dist = float('inf')
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    c1 = current_clusters[i]
                    c2 = current_clusters[j]
                    key = (min(c1, c2), max(c1, c2))
                    if key in distances:
                        dist = distances[key]
                        if dist < min_dist:
                            min_dist = dist
                            min_pair = (c1, c2)

            if min_pair is None:
                break
            # merge then delete the old cluster
            c1 ,c2 = min_pair
            clusters[c1].extend(clusters[c2])
            del clusters[c2]
            current_clusters.remove(c2)

            # update distnace with c1
            for c in current_clusters:
                if c != c1:
                    points_c1 = feature_space[clusters[c1]]
                    points_c = feature_space[clusters[c]]
                    new_dist = self.ward_distance(points_c1, points_c)
                    distances[(min(c1, c), max(c1, c))] = new_dist

        self.final_clusters = clusters # i know have the clusters
        self.centroids = [] # center of this clusters
        for cluster_idx in clusters:
            points = feature_space[clusters[cluster_idx]]
            centroid = np.mean(points, axis=0)
            self.centroids.append(centroid)
        self.centroids = np.array(self.centroids)
        print("finish agg")
        print(f"number of k is {len(self.centroids)}")
        print("Centroids RGB:", self.centroids[:, :3])


    def euclidean_distance(self, x1, x2): # this calulate the eculidain distance based on whole cluster
        return np.linalg.norm(x1 - x2)

    def get_idx_cloest_centroid(self, distances):
        return np.argmin(distances)

    def ward_distance(self, points_c1, points_c2):
        n1, n2 = len(points_c1), len(points_c2)
        if n1 == 0 or n2 == 0:
            return float('inf')
        centroid_c1 = np.mean(points_c1, axis=0)
        centroid_c2 = np.mean(points_c2, axis=0)
        squared_dist = np.sum((centroid_c1 - centroid_c2) ** 2)
        return (n1 * n2) / (n1 + n2) * squared_dist

    def apply_clustering_on_image(self):
        print("coloring original image...")
        img = self.input_viewer.current_image.modified_image
        h, w, _ = img.shape
        output_img = np.zeros((h, w, 3), dtype=np.uint8)


        print(f"number of final clusters: {len(self.final_clusters)}")
        print("cluster sizes:", [len(v) for v in self.final_clusters.values()])

        # map superpixels to final clusters
        superpixel_to_cluster = -np.ones(len(self.superpixel_centroids), dtype=int)  # Initialize with -1 for debugging

        for new_idx, (_, superpixels) in enumerate(self.final_clusters.items()):
            for sp in superpixels:
                if sp >= len(superpixel_to_cluster):
                    print(f"Warning: Superpixel index {sp} out of bounds!")
                    continue
                superpixel_to_cluster[sp] = new_idx


        unassigned = np.sum(superpixel_to_cluster == -1)
        if unassigned > 0:
            print(f"carning: {unassigned} superpixels not assigned to any cluster!")

        # Get cluster colors (RGB only) and make them more distinct
        cluster_colors = np.linspace(0, 255, num=self.num_of_clusters, dtype=np.uint8)
        cluster_colors = np.column_stack([cluster_colors,
                                          np.roll(cluster_colors, 1),
                                          np.roll(cluster_colors, 2)])


        print("assigned colors:")
        for i, color in enumerate(cluster_colors):
            print(f"cluster {i}: {color}")

        # color each pixels
        for y in range(h):
            for x in range(w):
                pixel_idx = y * w + x
                superpixel_idx = self.pixel_assignments[pixel_idx]

                if superpixel_idx >= len(superpixel_to_cluster):
                    print(f"Error: Pixel at ({x},{y}) has invalid superpixel index {superpixel_idx}")
                    continue

                cluster_idx = superpixel_to_cluster[superpixel_idx]

                if cluster_idx == -1:
                    print(f"Warning: Pixel at ({x},{y}) belongs to unassigned superpixel {superpixel_idx}")
                    cluster_idx = 0  # Default to first cluster

                if cluster_idx >= len(cluster_colors):
                    print(f"Error: Invalid cluster index {cluster_idx} for superpixel {superpixel_idx}")
                    cluster_idx = cluster_idx % len(cluster_colors)

                output_img[y, x] = cluster_colors[cluster_idx]

        self.output_viewer.current_image.modified_image = output_img
        print("Image coloring completed.")








