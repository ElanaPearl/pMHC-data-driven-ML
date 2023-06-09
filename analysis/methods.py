import numpy as np
from scipy.special import softmax, expit
from sklearn.neighbors import KDTree, KNeighborsClassifier
from cleanlab.rank import get_label_quality_scores
## Convention: Larger scores, more trustworthy predictions

class Entropy:
    def __init__(self):
        pass
    
    def fit(self, *args, **kwargs):
        pass
    
    def get_score(self, embeddings, logits, *args, **kwargs):
        return -expit(logits)*(1-expit(logits))


class TrustScore:
    """
    Code from https://github.com/google/TrustScore/blob/master/trustscore/trustscore.py
    Trust Score: a measure of classifier uncertainty based on nearest neighbors.
  """

    def __init__(self, k=10, alpha=0.0, filtering="none", min_dist=1e-12):
        """
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
        self.k = k
        self.filtering = filtering
        self.alpha = alpha
        self.min_dist = min_dist

    def filter_by_density(self, X: np.array):
        """Filter out points with low kNN density.

            Args:
            X: an array of sample points.

            Returns:
            A subset of the array without points in the bottom alpha-fraction of
            original points of kNN density.
            """
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        return X[np.where(knn_radii <= eps)[0], :]

    def filter_by_uncertainty(self, X: np.array, y: np.array):
        """Filter out points with high label disagreement amongst its kNN neighbors.

            Args:
            X: an array of sample points.
        
            Returns:
            A subset of the array without points in the bottom alpha-fraction of
            samples with highest disagreement amongst its k nearest neighbors.
            """
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        neigh.fit(X, y)
        confidence = neigh.predict_proba(X)
        cutoff = np.percentile(confidence, self.alpha * 100)
        unfiltered_idxs = np.where(confidence >= cutoff)[0]
        return X[unfiltered_idxs, :], y[unfiltered_idxs]

    def fit(self, embeddings, labels, *args, **kwargs):
        """Initialize trust score precomputations with training data.

                WARNING: assumes that the labels are 0-indexed (i.e.
                0, 1,..., n_labels-1).

                Args:
                X: an array of sample points.
                y: corresponding labels.
                """
        X  = embeddings
        y = labels
        self.n_labels = np.max(y) + 1
        self.kdtrees = [None] * self.n_labels
        if self.filtering == "uncertainty":
            X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
        for label in range(self.n_labels):
            if self.filtering == "none":
                X_to_use = X[np.where(y == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "density":
                X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "uncertainty":
                X_to_use = X_filtered[np.where(y_filtered == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)

            if len(X_to_use) == 0:
                print(
                    "Filtered too much or missing examples from a label! Please lower "
                    "alpha or check data."
                )

    def get_score(self, embeddings: np.array, predictions: np.array, *args, **kwargs):
        """Compute the trust scores.
        Given a set of points, determines the distance to each class.

        Args:
        X: an array of sample points.
        y_pred: The predicted labels for these points.

        Returns:
        The trust score, which is ratio of distance to closest class that was not
        the predicted class to the distance to the predicted class.
        """
        X  = embeddings
        y_pred = predictions
        d = np.tile(None, (X.shape[0], self.n_labels))
        for label_idx in range(self.n_labels):
            d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]
        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[range(d.shape[0]), y_pred]
        d_to_closest_not_pred = np.where(
            sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1]
        )
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)


class KNNConfidence:
    """Baseline which uses disagreement to kNN classifier.
  """

    def __init__(self, k=10):
        self.k = k

    def fit(self, embeddings, labels, *args, **kwargs):
        X = embeddings
        y = labels
        self.kdtree = KDTree(X)
        self.y = y

    def get_score(self, embeddings, predictions, *args, **kwargs):
        X = embeddings
        y_pred = predictions
        knn_idxs = self.kdtree.query(X, k=self.k)[1]
        knn_outputs = self.y[knn_idxs]
        return np.mean(
            knn_outputs == np.transpose(np.tile(y_pred, (self.k, 1))), axis=1
        )

class LabelQuality:
    """Baseline which uses disagreement to kNN classifier.
  """

    def __init__(self, method="confidence_weighted_entropy"):
        """
        Potential methods are
        confidence_weighted_entropy
        normalized_margin
        self_confidence
        """
        self.method = method

    def fit(self, embeddings, labels, logits, predictions, *args, **kwargs):
        return get_label_quality_scores(labels=labels,
                                        pred_probs=expit(logits),
                                        method=self.method)
    
    def get_score(self, embeddings, labels, logits, predictions, *args, **kwargs):
        return get_label_quality_scores(labels=labels,
                                        pred_probs=expit(logits),
                                        method=self.method)
