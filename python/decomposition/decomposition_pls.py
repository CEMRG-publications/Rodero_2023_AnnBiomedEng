from IO import DataHandler
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import os
from pathlib import Path


class PLSBinaryClassification(DataHandler):

    def __init__(self, dataset_path=None, dataset_filename=None, number_of_components=None, X=None, y=None,
                 percentiles=(10, 90)):
        super().__init__(dataset_path, dataset_filename, X=X, y=y)
        self.dataset_path = dataset_path
        self.dataset_filename = dataset_filename
        self.percentiles = percentiles
        self.number_of_components = number_of_components if number_of_components is not None else min(self.X.shape)
        self.count_balance = 0
        self.X_centered = np.zeros(self.X.shape)
        self.W = np.zeros((self.X.shape[1], self.number_of_components))
        self.T = np.zeros((self.X.shape[0], self.number_of_components))
        self.P = np.zeros((self.number_of_components, self.X.shape[1]))
        self.q = np.zeros((self.number_of_components, 1))
        self.E, self.f, self.b = (None,) * 3
        self.scaled_remodelling_components = np.zeros((self.number_of_components*len(self.percentiles),
                                                       self.number_of_components))

    @ staticmethod
    def get_pls_factors_vectors_binary(covariates, response):
        x_current = covariates  # could be original or residual
        y_current = response    # could be original or residual

        w = (x_current.T @ y_current).reshape(-1, 1)  # weight vector
        w /= np.linalg.norm(w)  # scaled weight vector
        t = x_current @ w   # score vector
        t_squared_norm = np.sum(np.square(t))
        p = t.T @ x_current / t_squared_norm  # X loadings
        q = y_current.T @ t / t_squared_norm  # y loading (scalar)

        x_resid = x_current - t @ p  # residual data matrix
        y_resid = y_current - t @ q  # residual response vector

        return w, t, p, q, x_resid, y_resid

    def get_pls_factors_binary(self, _x_centered, _y):
        x_current = _x_centered
        y_current = _y

        for component in range(self.number_of_components):
            _w, _t, _p, _q, x_current, y_current = self.get_pls_factors_vectors_binary(x_current, y_current)

            self.W[:, component] = _w.squeeze()  # weights/ BETA/ remodelling components
            self.T[:, component] = _t.squeeze()  # scores/ SCORE/ remodelling scores
            self.P[component, :] = _p.squeeze()  # X loadings
            self.q[component] = _q  # y loadings

        self.E = x_current  # Residual X
        self.f = y_current  # residual y
        self.b = self.W @ np.linalg.inv(self.P @ self.W) @ self.q + self.f  # relationship between X and y

    def get_predictions_binary(self, samples):
        estimation = []
        for sample in samples:
            estimation.append(1 if sample @ self.b >= 0 else -1)
        return estimation

    def assign_proper_labels_binary(self, class_mask):
        pos = self.y == class_mask[0]
        neg = self.y == class_mask[1]
        self.y[pos], self.y[neg] = 1, -1

    def get_class_balance_binary(self):
        classes, counts = np.unique(self.y, return_counts=True)
        assert len(classes) == 2, 'There must be 2 classes in this decomposition. Check your inputs.'

        # No matter how classes are encoded, they are turned to 1 and -1
        self.assign_proper_labels_binary(class_mask=classes)

        # To ensure proper boundary, centering must we weighted according to the balance of the classes
        _counts_ratio = counts[0] / counts[1]
        self.count_balance = (_counts_ratio - 1) / (_counts_ratio + 1)

    def decompose_with_pls(self, method='da'):
        self.get_class_balance_binary()  # check the balance between classes

        # Centering the covariates
        _X_mu = (np.mean(self.X[self.y == 1, :], axis=0) + np.mean(self.X[self.y == -1, :], axis=0)) / 2
        self.X_centered = self.X - _X_mu
        assert np.all(self.count_balance * np.mean(self.X_centered[self.y == 1, :], axis=0) -
                      np.mean(self.X_centered, axis=0) <= 10.0e-9), \
            'Classes are not centered properly. Check X centering rules.\n {}' \
                .format(self.count_balance * np.mean(self.X_centered[self.y == 1, :], axis=0) -
                        np.mean(self.X_centered, axis=0))

        if method == 'da':
            self.get_pls_factors_binary(self.X_centered, self.y)
        elif method == 'scikit':
            plsr = PLSRegression(self.number_of_components, scale=False)
            plsr.fit(self.X_centered, self.y)

    def get_weighted_component(self, _lambda=np.array(1), component=0):
        return _lambda * self.W[component, :]  # _lambda usually comes from score distribution

    def get_percentiles_of_pls_scores(self):
        score_percentiles = np.zeros((self.number_of_components, len(self.percentiles)))
        for component in range(self.number_of_components):
            for p_i, percentile in enumerate(self.percentiles):
                score_percentiles[component, p_i] = np.percentile(self.T[:, component], percentile) \
                                                    - np.mean(self.T[:component])
        return score_percentiles

    def get_remodelling_components(self):
        per = self.get_percentiles_of_pls_scores()
        per_len = len(self.percentiles)
        for component in range(self.number_of_components):
            self.scaled_remodelling_components[per_len*component:per_len*component+per_len, :] = \
                per[component, :].T.reshape(-1, 1) @ self.W[component, :].T.reshape(1, -1)

    def save_remodelling_components(self):
        self.save_result('remodelling_components.csv', self.scaled_remodelling_components)

    def save_transformed_data(self):
        self.save_result('scores.csv', self.T)


# -----PLS testing--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    path_to_data = os.path.join(str(Path.home()), 'Deformetrica', 'deterministic_atlas_ct',
                                'output_separate_tmp10_def10_prttpe13_corrected', 'Decomposition')
    data_filename = 'Momenta_Table.csv'

    data, target = load_iris(return_X_y=True)
    data = data[0:80, 0:3]
    target = target[0:80]
    pls = PLSBinaryClassification(dataset_filename=data_filename, dataset_path=path_to_data, X=data, y=target)
    pls.decompose_with_pls(method='da')

    plsr = PLSRegression(3, scale=False)
    x_plsr, y_plsr = plsr.fit_transform(pls.X_centered, pls.y)

    plt.scatter(plsr.x_scores_[pls.y == 1, 0], plsr.x_scores_[pls.y == 1, 1], c='red', marker='d')
    plt.scatter(plsr.x_scores_[pls.y == -1, 0], plsr.x_scores_[pls.y == -1, 1], c='blue', marker='x')
    x = np.linspace(-2, 2, 100)

    print('W:\n {}'.format(pls.W))
    print('xw:\n {}'.format(plsr.x_weights_))
    print('T:\n {}'.format(pls.T))
    print('Xload:\n {}'.format(plsr.x_loadings_.T @ plsr.x_loadings_))
    print('P:\n {}'.format(pls.P))
    print('q:\n {}'.format(pls.q))
    print('----------------')

    print('yload:\n {}'.format(plsr.y_loadings_))

    print('yw:\n {}'.format(plsr.y_weights_))
    print('x_scores:\n {}'.format(plsr.x_scores_))
    print('y_scores:\n {}'.format(plsr.y_scores_))
    print('xr:\n {}'.format(plsr.x_rotations_))
    print('yr:\n {}'.format(plsr.y_rotations_))
    pls.T -= np.mean(pls.T, axis=0)
    for i in range(3):
        print('T:\n min: {}, median: {}, mean:{}, max: {}'.format(np.min(pls.T[:, i]), np.median(pls.T[:, i]),
                                                                  np.mean(pls.T[:, i]), np.max(pls.T[:, i])))

    plt.plot(x, np.mean(pls.b[2])*x, 'k.-', linewidth=4)
    plt.plot(x, plsr.coef_[2]*x, 'y.-')
    plt.show()
