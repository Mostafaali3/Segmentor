from enum import Enum
class SegmentationType(Enum):
    KNN = "K-nearest neighbor (KNN)"
    MEAN_SHIFT = "Mean shifting"
    REGION_GROWING = "Region growing"
    AGGLOMERATIVE = "Agglomerative"