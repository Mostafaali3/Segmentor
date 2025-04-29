from enum import Enum
class SegmentationType(Enum):
    KNN = "K-means"
    MEAN_SHIFT = "Mean shifting"
    REGION_GROWING = "Region growing"
    AGGLOMERATIVE = "Agglomerative"