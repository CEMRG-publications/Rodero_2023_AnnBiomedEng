Code used for the paper [Rodero C, Longobardi S, Augustin C, Strocchi M, Plank G, Lamata P, Niederer SA. Calibration of
Cohorts of Virtual Patient Heart Models Using Bayesian History Matching. Annals of Biomedical Engineering. 2022
Oct 21:1-2](https://pubmed.ncbi.nlm.nih.gov/36271218/).

The folders submission_files and sh contain scripts for mechanics simulations that were eventually not run. A good place
to start is in python/ and the jupyter notebooks. Most of the functions are documented. The ones that are not were
probably not used in the final version of the paper.

Scripts or folders with the suffix "_old" can potentially be removed safely (I just didn't dare). When running the SSM,
make a copy of the folder CardiacMeshConstruction_outside first because there are hardcoded paths in some functions.
