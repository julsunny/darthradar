import numpy as np
import sklearn.cluster as skc
from sklearn import preprocessing
from sklearn.feature_extraction import image

#%%
def height_map(rdms):
    fm_rdms = []
    for i in range(rdms.shape[0]):
        for j in range(rdms.shape[1]):
            fm_rdms.append([i, j, float(rdms[i,j])])
    
    return np.asarray(fm_rdms)

def fm_t_data(rdms_raw, color_scaling = 1):
    """
    Applies feature mapping heightmap and normalizes colordimension. Color_scaling determines 
    the relative importance of coloring relative to position for clustering.

    Parameters
    ----------
    rdms_raw : TYPE
        DESCRIPTION.
    color_scaling : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    fmt_rdms : TYPE
        DESCRIPTION.

    """
    #apply feature mapping
    fmt_rdms = height_map(rdms_raw)
    #normalize color dimension
    fmt_rdms2 = preprocessing.scale(fmt_rdms[:,2], with_mean = False, axis = 0)
    fmt_rdms[:,2] = fmt_rdms2
    #scale relative importance of color dimension
    fmt_rdms[:,2] *= color_scaling
    
    return fmt_rdms

class Clustering:
    def __init__(self, clusters):
        self.clusters = np.asarray(clusters, dtype = Cluster)
    
    def get_centres(self):
        return np.asarray([c.mean for c in self.clusters])
    
    def append_cluster(self, new_cluster):
        self. clusters = np.concatenate((self.clusters, np.asarray([new_cluster], dtype = Cluster)), axis = None)
        
    def get_boxes(self):
        boxes = []
        for cluster in self.clusters:
            boxes.append(cluster.get_box())
        return np.asarray(boxes)

class Cluster:
    def __init__(self, c_points, label):
        self.label = label
        self.c_points = c_points
        self.mean = np.mean(c_points, axis = 0)
        self.var = np.var(c_points, axis = 0)
        self.freq = c_points.shape[0]
        self.box = self.get_box()
        
    def average_var(self):
        return np.mean(self.var)
    
    def get_box(self):
        x0 = np.min(self.c_points[:,0])
        y0 = np.min(self.c_points[:,1])
        x1 = np.max(self.c_points[:,0])
        y1 = np.max(self.c_points[:,1])
        
        return [x0, y0, x1, y1]

class DBSCAN():
    """A density based clustering method that expands clusters from 
    samples that have more neighbors within a radius specified by eps
    than the value min_samples.
    Parameters:
    -----------
    eps: float
        The radius within which samples are considered neighbors
    min_samples: int
        The number of neighbors required for the sample to be a core point. 
    """
    def __init__(self, eps=1, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def _get_neighbors(self, sample_i):
        """ Return a list of indexes of neighboring samples
        A sample_2 is considered a neighbor of sample_1 if the distance between
        them is smaller than epsilon """
        neighbors = []
        idxs = np.arange(len(self.X))
        for i, _sample in enumerate(self.X[idxs != sample_i]):
            distance = np.linalg.norm(self.X[sample_i] - _sample)
            if distance < self.eps:
                neighbors.append(i)
        return np.array(neighbors)

    def _expand_cluster(self, sample_i, neighbors):
        """ Recursive method which expands the cluster until we have reached the border
        of the dense area (density determined by eps and min_samples) """
        cluster = [sample_i]
        # Iterate through neighbors
        for neighbor_i in neighbors:
            print(len(self.visited_samples))
            if not neighbor_i in self.visited_samples:
                self.visited_samples.append(neighbor_i)
                # Fetch the sample's distant neighbors (neighbors of neighbor)
                self.neighbors[neighbor_i] = self._get_neighbors(neighbor_i)
                # Make sure the neighbor's neighbors are more than min_samples
                # (If this is true the neighbor is a core point)
                if len(self.neighbors[neighbor_i]) >= self.min_samples:
                    # Expand the cluster from the neighbor
                    expanded_cluster = self._expand_cluster(
                        neighbor_i, self.neighbors[neighbor_i])
                    # Add expanded cluster to this cluster
                    cluster += expanded_cluster

                else:
                    # If the neighbor is not a core point we only add the neighbor point
                    cluster.append(neighbor_i)
        return cluster
    
    def _get_id(self, x0):
        x = int(x0[0])
        y = int(x0[1])
        
        found = False
        for i, point in enumerate(self.X):
            if (int(point[0]) == x) and (int(point[1]) == y):
                found = True
                idx = i
                break
        
        assert found == True; "Points was not found."
        return idx
    
    def _get_cluster_labels(self):
        """ Return the samples labels as the index of the cluster in which they are
        contained """
        # Set default value to number of clusters
        # Will make sure all outliers have same cluster label
        labels = np.full(shape=self.X.shape[0], fill_value = 0)
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i + 1
        return labels

    # DBSCAN
    def predict(self, X, x0):
        self.X = X
        self.x0 = x0
        self.clusters = []
        self.visited_samples = []
        self.neighbors = {}
        self.id_x0 = self._get_id(x0)
        
        # Iterate through samples and expand clusters from them
        # if they have more neighbors than self.min_samples
        self.neighbors[self.id_x0] = self._get_neighbors(self.id_x0)
        if len(self.neighbors[self.id_x0]) >= self.min_samples:
            # If core point => mark as visited
            self.visited_samples.append(self.id_x0)
            # Sample has more neighbors than self.min_samples => expand
            # cluster from sample
            new_cluster = self._expand_cluster(
                self.id_x0, self.neighbors[self.id_x0])
            # Add cluster to list of clusters
            self.clusters.append(new_cluster)
        else:
            print("deacrease min_samples or increase epsilon")

        # Get the resulting cluster labels
        cluster_labels = self._get_cluster_labels()
        return cluster_labels
#%%

def refine_box(rdms_raw, pre_box, strength, n_objects = 1, gradient = True, color_scaling = None, clustering_kwargs = {"eigen_solver" : "arpack", "n_jobs" : -1}):
    #make copy of image
    rdms = rdms_raw[pre_box[0]:pre_box[2], pre_box[1]:pre_box[3]].copy()
    
    #some preprocessing, scale color dimension
    if color_scaling is not None:
        rdms_2_norm = preprocessing.scale(rdms[:,2], with_mean = False, axis = 0)
        rdms[:,2] = rdms_2_norm * color_scaling
    
    if gradient:
        graph = image.img_to_graph(rdms)
        graph.data = np.exp(-graph.data)
        
        SC = skc.SpectralClustering(affinity = "precomputed", n_clusters = n_objects**clustering_kwargs)
        
        SC.fit(graph)
        labels = SC.labels_
        
        clusters = [Cluster(height_map(rdms)[labels == l], l) for l in set(labels)]
        clustering = Clustering(clusters)
        
        boxes = clustering.get_boxes()
    
    else:
        fmt_rdms = fm_t_data(rdms, color_scaling = color_scaling)
        
        SC = skc.SpectralClustering(n_clusters = n_objects, **clustering_kwargs)
    
        SC.fit(fmt_rdms)
        labels = SC.labels_
        
        clusters = [Cluster(fmt_rdms[labels == l], l) for l in set(labels)]
        clustering = Clustering(clusters)
        
        boxes = clustering.get_boxes()
        
    return boxes
    
    
    
    
    
    
    
    
    
    
    
    
    
    