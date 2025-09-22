# Federated Learning MoDN
Model architectures and training examples. 

<mark>Important: everything related to "feature decoding" (i.e. reconstructing feature values from the state) does not work atm</mark>
<mark>Also, the encoders/decoders with "shift" can be ignored (idea was to have some layers with shared FL parameters and some layers trained only locally, but doesnt seem to work well)</mark>


## Main scripts
- ``modules_modn.py``: MoDN architecture
- ``modules_baseline.py``: MLP Baseline architecture
- ``training_example.py``: instantiates a MoDN model and trains it on simple data (no FL)
- ``federated_training_example.py``: FL training for MoDN or MLP baseline (depending on config file parameters) on different nodes. Number of nodes, samples, training parameters, etc can be modified in the ``config_data.yaml`` and ``config_training.yaml`` files
- ``epoct_training.py`` : FL training for MoDN on epoct+ data. The dataset ``epoct_plus_cleaned.csv`` has to be saved in a ``datasets`` folder (to be created) for the code to run


## Resources
- [MoDN] (https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000108)

- [FedMoDN] (https://openreview.net/pdf?id=Edn2EigJFP)

- [MultiMoDN] (https://proceedings.neurips.cc/paper_files/paper/2023/hash/5951641ad71b0052cf776f9b71f18932-Abstract-Conference.html)

### ePOCT+ dataset
- [Dataset] (https://www.nature.com/articles/s41591-023-02633-9)


