import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

import matplotlib.pyplot as plt

from args_parser import parse_args

"""
--- Do not delete this comment ---

This exercise is the final exercise of the course. It is based on the previous exercises we have done - use ex1, ex2, ex3 where it's relevant. You can copy code from there, and if any relevant exist there, do use them.
Information about the data:
- 160 trials were preformed, In each trial the subject was asked to imagine motor activity in one of the 2 classes: left hand or right hand.
- For each trial, 2 EEG channels (C3 and C4) were recorded.
- The trials are split into training (128 labeled trials in) and testing (32 unlabeled trials) sets. The training data is stored in 𝑑𝑎𝑡𝑎/𝑚𝑜𝑡𝑜𝑟_𝑖𝑚𝑎𝑔𝑒𝑟𝑦_𝑡𝑟𝑎𝑖𝑛_𝑑𝑎𝑡𝑎. 𝑛𝑝𝑦 and the test data at 𝑑𝑎𝑡𝑎/𝑚𝑜𝑡𝑜𝑟_𝑖𝑚𝑎𝑔𝑒𝑟𝑦_𝑡𝑒𝑠𝑡_𝑑𝑎𝑡𝑎. 𝑛𝑝𝑦, both as 𝑛𝑢𝑚𝑝𝑦 arrays.
- The sampling rate is stored in the training file.

The acquired data are stored in the data field, of dimensions 128 x 768 x 3. The first dimension represents the trials, the second dimension represents time samples and the third dimension represents the channels. Channel 1 is electrode C3, Channel 2 is electrode C4 and Channel 3 is a trigger channel (not relevant for this project).

The correct labels of each trial are found in the training file, in dimensions 4 x 128. The second dimension represents the trials. The first dimension (rows) represents 4 possible labels (see the 𝑙𝑎𝑏𝑒𝑙𝑠_𝑛𝑎𝑚𝑒 field): ARTIFACT (1st row), REMOVE (2nd row), LEFT (3rd row), RIGHT (4th row). Only use trials where the attribute is LEFT or RIGHT.
Assumptions:
- There is an equal number of left and right trials (you can assume it when you write code).
- The 𝑡𝑒𝑠𝑡𝑖𝑛𝑔 data contains just a 32 x 768 x 3 data matrix.
- All the data has already gone through pre-processing and does not require further cleaning

Our goal is to extract useful features from the data and then train a classification algorithm on these features to predict the label (left or right) on a single trial basis.

First, we will explore the data and search for informative features. Once you identify such features, we'll extract them and build our classifier. Once your classifier is ready, fiddle with the features to improve the classifier accuracy. Finally, test the performance of your code with the testing set.

General Guidelines:
* All before each visualization, I want to have a comment describing the exact graph showing, which would contribute to the report. Mark these comments with a unique report tag, so it'll be easy to tell where they are.
* Use the report tags in descriptions that would contribute to understanding of the general process/code, which will go to the report as well.
* * keep the comments short and concise
* * Not all comments are for the report, pick them wisely.
* Doc string should have a blank line after \"\"\" and before the text

Additional info:
* Dataset: g.BSanalyze testdata "Right and Left Hand Movement Imagination" (user manual p.319, testdata/RightLeftImagination).
- Contralateral channel mapping: C3 = right-hand movement representation area
  (left hemisphere); C4 = left-hand movement representation area (right
  hemisphere). I.e. right-hand imagery -> C3, left-hand imagery -> C4.
- Cue/timing (shared "Movement Imagination" timeline, 8 s trial): fixation
  cross at 0 s, beep ~2 s, arrow cue 3 s -> 4.25 s (arrow right -> imagine
  right hand, arrow left -> imagine left hand), imagination until 8 s.
- Volume: 4 runs x 40 trials = 160 trials. Derivation: referenced to right ear.
- Timing nuance (assumption): manual describes 8 s trials, but our .npz epochs
  are 768 samples = 6 s at 128 Hz, so the provided data is likely a cropped
  sub-window of the full 8 s trial.


Q1: Visualization

Define C3_COLOR and C4_COLOR. Also define RIGHT_COLOR and LEFT_COLOR. Make sure the 4 colors are distinct. (Use blue, red, green, yellow)
- Load the training data.
- Visualize the EEG signal in a single channel for trials from a single class.
- - For each class (left and right) draw a figure with 20 subplots corresponding to 20 random trials.
- - Use C3_COLOR and C4_COLOR.
- - Each subplot should plot the data from both channels (C3 and C4).
- Eyeball the data and see if you can identify qualitative differences between the
different classes.

Q2: Spectral analysis

A Power spectrum -- use previous existing code if possible
- Calculate the power spectrum from all trials in each class.
- - Use Welch’s method (𝑠𝑐𝑖𝑝𝑦. 𝑠𝑖𝑔𝑛𝑎𝑙. 𝑤𝑒𝑙𝑐ℎ function).
- - Average across trials’ spectra.
- - If you are using time windows, make sure they are at least 0.5 𝑠𝑒𝑐𝑜𝑛𝑑𝑠 long (preferably over 1 𝑠𝑒𝑐𝑜𝑛𝑑) to allow for a good estimation of the power in low frequencies.
- - DO NOT use samples from the entire trial, remember that motor imagery took place only in a specific time segment (see 𝑚𝑜𝑡𝑜𝑟 𝑖𝑚𝑎𝑔𝑒𝑟𝑦 𝑑𝑎𝑡𝑎 pdf).
- - You are allowed to call welch once per channel.
- Plot a spectrum for each class and each channel (a total of 4 spectra). Create one figure with 2 subplots (one for each channel), where each subplot contains 2 spectra (for each class).
- - Use 𝑅𝐼𝐺𝐻𝑇_𝐶𝑂𝐿𝑂𝑅 and 𝐿𝐸𝐹𝑇_𝐶𝑂𝐿𝑂𝑅.
- - Add a 1-standard-error confidence interval to the plots.
- Compare the power spectra of both classes: Are there any frequency bands that seem useful for separating the classes? Describe in your report.
- - Do this by analyzing the plots.



"""


## ====================================================================
## ============================ Parameters ============================
## ====================================================================

DATA_PATH = Path(parse_args().data_dir)
TRAIN_DATA_PATH = DATA_PATH / 'motor_imagery_train_data.npz'
TEST_DATA_PATH = DATA_PATH / 'motor_imagery_test_data.npz'

## ====================================================================
## ========================= 1: Visualization =========================
## ====================================================================

# [REPORT] Four distinct colors are used consistently across all visualizations:
# the two EEG channels (C3, C4) are encoded by C3_COLOR / C4_COLOR, and the two
# imagined-movement classes (LEFT, RIGHT) by LEFT_COLOR / RIGHT_COLOR. All four
# colors (blue, red, green, yellow) are visually distinct.
C3_COLOR = 'blue'
C4_COLOR = 'red'
LEFT_COLOR = 'green'
RIGHT_COLOR = 'yellow'

# ---------------------------------------------------------------------
N_SAMPLES = 768          # time samples per trial
# [REPORT] The archive does NOT contain a sampling-rate field even though the
# task text says it should. For this motor-imagery recording each trial spans
# 768 samples; the standard sampling rate for this dataset is 128 Hz, giving a
# trial length of 768 / 128 = 6 seconds. We use FS = 128 Hz throughout.
FS = 128                 # sampling rate [Hz]
DT = 1 / FS              # time step [sec]

# Channel indices within the data array
C3_IDX = 0               # electrode C3
C4_IDX = 1               # electrode C4

# Label-matrix row indices (0-indexed) for the two classes we use
LEFT_ROW = 2             # 3rd row -> LEFT
RIGHT_ROW = 3            # 4th row -> RIGHT

N_PLOT_TRIALS = 20       # number of random trials drawn per class figure

# Make the random trial selection reproducible for the report - choose a more interesting seed
rng = np.random.default_rng(0)

train_npz = np.load(TRAIN_DATA_PATH, allow_pickle=True)
train_data = train_npz['data']        # (128, 768, 3)
train_labels = train_npz['labels']    # (4, 128)

# [REPORT] Build the time vector (in seconds) shared by every trial: 768 samples
# sampled at 128 Hz spans 0 .. ~6 s. This is the x-axis for all signal plots.
time_vector = np.arange(N_SAMPLES) * DT   # [sec]

left_trial_idx = np.where(train_labels[LEFT_ROW] == 1)[0]
right_trial_idx = np.where(train_labels[RIGHT_ROW] == 1)[0]


def plot_class_trials(class_name, class_trial_idx):
    """
    Draw one figure of N_PLOT_TRIALS subplots for a single class.

    Each subplot shows one randomly chosen trial of the class, plotting both EEG
    channels: C3 in C3_COLOR and C4 in C4_COLOR, against the shared time vector.
    """
    # Randomly choose N_PLOT_TRIALS distinct trials from this class
    n_available = len(class_trial_idx)
    chosen = rng.choice(class_trial_idx, size=min(N_PLOT_TRIALS, n_available),
                        replace=False)

    # 4 rows x 5 cols = 20 subplots
    fig, axes = plt.subplots(4, 5, figsize=(18, 12), sharex=True)
    fig.suptitle(f'{class_name} class - 20 random trials (C3 vs C4)',
                 fontsize=14, fontweight='bold')

    for ax, trial in zip(axes.ravel(), chosen):
        ax.plot(time_vector, train_data[trial, :, C3_IDX],
                color=C3_COLOR, linewidth=0.6, label='C3')
        ax.plot(time_vector, train_data[trial, :, C4_IDX],
                color=C4_COLOR, linewidth=0.6, label='C4')
        ax.set_title(f'Trial {trial}', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Shared axis labels and a single legend for the whole figure
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Amplitude')
    handles, plot_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, plot_labels, loc='upper right')

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# [REPORT] Figure 1 (LEFT class): a 4x5 grid of 20 subplots, each showing one
# randomly selected LEFT-hand motor-imagery trial. In every subplot the C3
# electrode is drawn in blue and the C4 electrode in red, against time in
# seconds. This lets us eyeball qualitative within-class structure and later
# compare it against the RIGHT class.
plot_class_trials('LEFT', left_trial_idx)

# [REPORT] Figure 2 (RIGHT class): the same 4x5 grid of 20 subplots for 20
# randomly selected RIGHT-hand motor-imagery trials, with C3 in blue and C4 in
# red. Comparing this figure with the LEFT figure is the qualitative inspection
# step requested in Q1 (looking for class-dependent differences between C3/C4).
plot_class_trials('RIGHT', right_trial_idx)

plt.show()

"""
[REPORT]
Q1 - Qualitative observations (eyeball inspection):

Based on visually inspecting the two rendered figures (20 random LEFT trials and
20 random RIGHT trials, C3 in blue / C4 in red) plus a mean+-std overlay:

* Both classes look like broadband, zero-mean noisy EEG: fast oscillations
  fluctuating. The overall morphology of LEFT and RIGHT trials is very similar to the eye.
* In every subplot C3 (blue) and C4 (red) overlap heavily and track each other
  closely; the two channels are NOT visibly de-correlated within a trial.
* The only mild, repeatable tendency is that C4 (red) shows slightly larger
  amplitude / peak-to-peak excursions than C3, and this is a touch more
  pronounced in RIGHT trials (a few RIGHT trials reach ~+-20). This is subtle
  and swamped by trial-to-trial variability.
* In the mean+-std overlay the trial-averaged traces sit near zero with almost
  fully overlapping std bands across classes - no class-level structure is
  visible at the average level.

Visual separability: the classes are NOT cleanly separable by eye in the raw
time-domain signal; within-class variability dominates - so raw amplitude is a
weak feature and the discriminative info likely lives in the frequency domain
(mu/beta band power, C3/C4 lateralization), to be quantified in Q2+.
"""

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
max_n_pcs = None        # TODO (Q4): set max number of PCA components
max_n_features = None   # TODO (Q4): set max number of features

## ====================================================================
## ======================== 5: Model testing ==========================
## ====================================================================

RESULTS_PATH = None  # TODO (Q5): Path OBJECT to the results folder
