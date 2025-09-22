# Federated Learning MoDN
Model architectures and training examples. 

<mark>Important: everything related to "feature decoding" (i.e. reconstructing feature values from the state) does not work atm</mark>
<mark>Also, the encoders/decoders with "shift" can be ignored (idea was to have some layers with shared FL parameters and some layers trained only locally, but doesnt seem to work well)</mark>


## Main scripts
- ``modules_modn.py``: MoDN architecture
- ``modules_baseline.py``: MLP Baseline architecture
- ``training_example.py``: instantiates a MoDN model and trains it on simple data (no FL)
- ``federated_training_example.py``: FL training for MoDN or MLP baseline (depending on config file parameters) on different nodes. Number of nodes, samples, training parameters, etc can be modified in the ``config_data.yaml`` and ``config_training.yaml`` files
- ``epoct_training.py`` : FL training for MoDN on epoct+ data. The dataset 'epoct_plus_cleaned.csv' has to be saved in the 'datasets' folder for the code to run

## ePOCT+ dataset
- [Paper] (https://www.nature.com/articles/s41591-023-02633-9)
- ``epoct_plus_cleaned.csv``: pre-processed data from the rct described in the nature paper. Only patients from "intervention" arm. Features and targets have been selected based on medical relevance and prevalence. 
- ``epoct_plus_cleaned_report.html`` : dataset profile report

