# Decomposition
## A framework for linear dimensionality reduction
The decomposition framework contains a set of tools for easy and fast application of linear analysis of single- and multi-dimensional shapes. Statistical shape analysis allows for the determination of dominant features in the data set of similar point clouds in multidimensional space. In the clinical setting, it is used to find the anatomical biomarkers of adaptive or maladaptive remodelling, which can serve as survival predictors or disease severity indicators.
### Principal component analysis
PCA allows to compute the modes of maximum variance in the set of shapes. The core of the method is the calculation of the eigenvectors and eigenvalues of the covariance matrix build from shape descriptors. The orthogonal eigenvectors describe the directions of the modes, while corresponding eigenvalues determine the explained variance. Once computed, each subject used in the calculation may be assigned a Z-score along all the most relevant modes and used for the classification/regression purposes.
### Partial least squares regression
TODO
### Supervised principal component analysis
TODO
### Information maximization component analysis
TODO???
