import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_eeglab
from mne.decoding import CSP

