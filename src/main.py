import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

from scipy.signal import welch

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
* Add a unique figure subtag before describing figures
* Use the report tags in descriptions that would contribute to understanding of the general process/code, which will go to the report as well.
* * keep the comments short and concise
* * Not all comments are for the report, pick them wisely.
* Doc string should have a blank line after \"\"\" and before the text
* All temp files should be in the temp/ folder.
* * The temp/ folder should be deleted at the end of the entire project

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
# colors (blue, red, green, orange) are visually distinct.
C3_COLOR = 'blue'
C4_COLOR = 'red'
LEFT_COLOR = 'green'
RIGHT_COLOR = 'orange'

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

# Shared training data + derived arrays, reused across questions (Q1, Q2, ...)
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


def run_q1():
    """

    Render the Q1 per-class trial grids (LEFT and RIGHT) and show them.
    """
    # [REPORT][FIG-Q1-LEFT] Figure 1 (LEFT class): a 4x5 grid of 20 subplots, each
    # showing one randomly selected LEFT-hand motor-imagery trial. In every subplot
    # the C3 electrode is drawn in blue and the C4 electrode in red, against time in
    # seconds. This lets us eyeball qualitative within-class structure and later
    # compare it against the RIGHT class.
    plot_class_trials('LEFT', left_trial_idx)

    # [REPORT][FIG-Q1-RIGHT] Figure 2 (RIGHT class): the same 4x5 grid of 20 subplots
    # for 20 randomly selected RIGHT-hand motor-imagery trials, with C3 in blue and
    # C4 in red. Comparing this figure with the LEFT figure is the qualitative
    # inspection step requested in Q1 (looking for class differences between C3/C4).
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

# [REPORT] Motor-imagery sub-window. The g.BSanalyze "Movement Imagination"
# timeline is an 8 s trial: fixation at 0 s, beep ~2 s, arrow cue 3 -> 4.25 s,
# and the subject keeps imagining the movement until 8 s. Our .npz epoch is only
# 768 samples = 6 s at 128 Hz, i.e. a cropped sub-window of that 8 s trial. To
# capture the imagery (and avoid the pre-cue baseline/fixation), we analyse the
# second half of the epoch, from 3.0 s to 6.0 s. At FS = 128 Hz this is samples
# [384, 768), a 3 s window (384 samples) -- well over the >=1 s recommended for
# stable low-frequency (mu ~8-13 Hz) estimation.
IMAGERY_START_S = 3.0                          # imagery window start [sec]
IMAGERY_END_S = 6.0                            # imagery window end [sec]
IMAGERY_START_IDX = int(IMAGERY_START_S * FS)  # -> sample 384
IMAGERY_END_IDX = int(IMAGERY_END_S * FS)      # -> sample 768

# [REPORT] Welch parameters: nperseg = 1.0 s (128 samples) gives ~1 Hz frequency
# resolution, satisfying the ">=0.5 s, preferably >=1 s" requirement; 50% overlap
# (Hann window) yields several segments inside the 3 s window for a smoother PSD.
WELCH_NPERSEG = min(int(FS * 1.0), IMAGERY_END_IDX - IMAGERY_START_IDX)
WELCH_NOVERLAP = WELCH_NPERSEG // 2

# Restrict the displayed spectrum to a sensible EEG range so mu (~8-13 Hz) and
# beta (~13-30 Hz) are visible without high-frequency clutter.
SPECTRUM_FREQ_RANGE = (0, 40)                  # [Hz]


def class_channel_psd(class_trial_idx, channel_idx):
    """

    Compute the trial-averaged Welch PSD for one class on one channel.

    Slices every trial of the class to the motor-imagery window, runs welch once
    (passing the stacked trials as a 2D array so welch returns a PSD per trial),
    then averages across trials and returns the +-1 standard-error band.
    Trial data is passed in explicitly. Returns: f, p_mean, p_sem.
    """
    # (n_trials, window_len): one row per trial, restricted to the imagery window
    segments = train_data[class_trial_idx, IMAGERY_START_IDX:IMAGERY_END_IDX,
                          channel_idx]
    # One welch call per channel/class; axis=-1 -> a PSD per trial (per row).
    f, psd_per_trial = welch(segments, fs=FS, nperseg=WELCH_NPERSEG,
                             noverlap=WELCH_NOVERLAP, window='hann',
                             scaling='density', detrend=False, axis=-1)
    p_mean = psd_per_trial.mean(axis=0)
    n_trials = psd_per_trial.shape[0]
    # +-1 standard ERROR across trials (not std): std / sqrt(n).
    p_sem = psd_per_trial.std(axis=0, ddof=1) / np.sqrt(n_trials)
    return f, p_mean, p_sem


def plot_psd_on_ax(ax, f, p, p_sem, color, label):
    """

    Plot a single PSD trace on `ax` (in dB) with a +-1 SE shaded band.

    Adapted from ex3's plot_psd_on_ax: the shaded band is exactly +-1 standard
    error (not a 95% CI), and the trace is shown in dB (10*log10).
    """
    mask = (f >= SPECTRUM_FREQ_RANGE[0]) & (f <= SPECTRUM_FREQ_RANGE[1])
    fz, pz, semz = f[mask], p[mask], p_sem[mask]
    p_db = 10 * np.log10(pz)
    # Propagate the SE through the dB transform (delta method).
    sem_db = 10 / np.log(10) * (semz / pz)
    ax.plot(fz, p_db, color=color, label=label, lw=1.4)
    ax.fill_between(fz, p_db - sem_db, p_db + sem_db, color=color, alpha=0.25,
                    linewidth=0, label=f'{label} +-1 SE')


def run_q2():
    """

    Render the Q2 power-spectrum figure (C3/C4 subplots, LEFT vs RIGHT) and show it.
    """
    # [REPORT][FIG-Q2-SPECTRA] Figure 3 (power spectra): one figure with two subplots,
    # C3 (left) and C4 (right). Each subplot overlays the trial-averaged Welch power
    # spectrum of the LEFT class (LEFT_COLOR) and the RIGHT class (RIGHT_COLOR), each
    # with a +-1 standard-error band, computed only on the 3-6 s motor-imagery window.
    # PSD is shown in dB over 0-40 Hz so the mu (~8-13 Hz) and beta (~13-30 Hz)
    # bands are visible for comparing class separability.
    fig_psd, (ax_c3, ax_c4) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig_psd.suptitle('Power spectra (Welch, 3-6 s imagery window) by channel & class',
                     fontsize=14, fontweight='bold')

    for ax, ch_idx, ch_name in ((ax_c3, C3_IDX, 'C3'), (ax_c4, C4_IDX, 'C4')):
        f_left, p_left, sem_left = class_channel_psd(left_trial_idx, ch_idx)
        f_right, p_right, sem_right = class_channel_psd(right_trial_idx, ch_idx)
        plot_psd_on_ax(ax, f_left, p_left, sem_left, LEFT_COLOR, 'LEFT')
        plot_psd_on_ax(ax, f_right, p_right, sem_right, RIGHT_COLOR, 'RIGHT')
        ax.set_title(ch_name)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim(SPECTRUM_FREQ_RANGE)
        ax.grid(alpha=0.3)
        ax.legend()
    ax_c3.set_ylabel('PSD [dB re µV²/Hz]')

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


# [REPORT] Candidate frequency bands for class separation. The mu (8-13 Hz) and beta (13-30 Hz) sensorimotor rhythms
# are the classic event-related (de)synchronization bands for motor imagery.
info_bands = {'mu': (8, 13), 'beta': (13, 30)}

"""
[REPORT]
Q2 - Power-spectrum comparison (grounded in the rendered figure):

* The single most separating feature is a sharp RIGHT-class beta peak at
  ~16-17 Hz that is essentially absent in the LEFT class on BOTH channels.
  It is strongest on C4 (RIGHT rises ~6 dB above LEFT, well outside the
  narrow +-1 SE bands -> a real, not noise-driven, difference) and clearly
  present on C3 too (RIGHT ~1 dB vs LEFT ~-9 dB at the peak).
* Low beta (13-30 Hz) is therefore the most useful band: averaged over the
  band, RIGHT has markedly higher power than LEFT (C4: ~6 dB, C3: ~4 dB).
* In the mu band (8-13 Hz) the difference is weaker and reversed: LEFT shows
  a slightly higher mu bump than RIGHT (~1-2 dB, C3 a bit more than C4), so
  mu is a secondary, weaker discriminator.
* Contralateral pattern (C3<->right hand, C4<->left hand): only partially
  visible. The beta separation appears on both channels rather than being
  cleanly lateralised, and if anything the right-hand (RIGHT) beta increase
  is most prominent on C4 (the left-hand area), so the textbook contralateral
  ERD/ERS mapping is NOT clean in this data.
* Conclusion: beta-band (~13-30 Hz, especially the ~16 Hz peak) power is the
  promising feature for LEFT-vs-RIGHT separation, with mu as a weaker backup;
  both are carried forward in info_bands for Q3 feature extraction.
"""

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


def main():
    """

    Run every question's analysis in order.
    """
    run_q1()
    run_q2()


if __name__ == '__main__':
    main()
