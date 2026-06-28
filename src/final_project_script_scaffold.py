import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

import matplotlib.pyplot as plt

from args_parser import parse_args

## ====================================================================
## ============================ Parameters ============================
## ====================================================================

DATA_PATH = Path(parse_args().data_dir)
TRAIN_DATA_PATH = DATA_PATH / 'motor_imagery_train_data.npy'
TEST_DATA_PATH = DATA_PATH / 'motor_imagery_test_data.npy'

## ====================================================================
## ========================= 1: Visualization =========================
## ====================================================================

C3_COLOR = 
C4_COLOR = 
LEFT_COLOR =
RIGHT_COLOR =

## ====================================================================
## ======================= 2: Spectral Analysis =======================
## ====================================================================

info_bands = {}

## ====================================================================
## ====================== 3: Feature Extraction =======================
## ====================================================================

df = pd.DataFrame()

def band_power(psd, f_min, f_max):
    pass


## ====================================================================
## ======================== 4: Classification =========================
## ====================================================================

min_n = 2
max_n_pcs =
max_n_features =

## ====================================================================
## ======================== 5: Model testing ==========================
## ====================================================================

RESULTS_PATH = # TODO: Path OBJECT to the results folder
