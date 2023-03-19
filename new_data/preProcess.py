import mne
import numpy as np
import pandas as pd


'''
Preprocess the EEG data, including epoching, filtering, and labeling the epoch with corresponding
labels(left hand, right hand, feet). One hot encode the labels with pandas (convert np.array into df first)
for random forest afterward. 
'''
class preProcess:
    # Read the edf file.
    def __init__(self, file):
        self.file = file
        # file = "\\UCSD\\2023 winter semester\\COGS 189\\final project\\sourcedata\\rawdata\\S001\\S001R03.edf"
        self.raw = mne.io.read_raw_edf(file)
        self.raw_data = self.raw.get_data()
        # get raw data info.
        self.info = self.raw.info
        # get parameters respectively.
        self.channels = self.raw.ch_names
        self.n_time_samps = self.raw.n_times
        self.time_secs = self.raw.times
        self.ch_names = self.raw.ch_names
        self.sampling_freq = self.raw.info['sfreq']
        # visualize original dataset.
        # raw.plot(duration=60, start=0, scalings=dict(eeg=1e-4), n_channels=64, title='Raw signal', block=True)

    def epoch(self):
        # get events.
        event_time, event_dict = mne.events_from_annotations(self.raw)
        # get the event list without T(1)
        target_event = [x for x in event_time if x[2] != 1]
        print(target_event)

        # to be filtered

        # epoch data to range (-0.2, 0.8). Magic number to be fixed.
        epoch_raw = mne.Epochs(self.raw, target_event, tmin=-0.2, tmax=0.8, baseline=None)
        epoch_data = epoch_raw.get_data()
        print(epoch_data)
        # visualize data after epoch.
        # epoch_raw.plot(block=True, scalings=dict(eeg=1e-4))

    '''
    filter and normalize the data
    '''
    def filter(self):
        # show graph and then try to use high-pass/low-pass

        # file = "/Users/lijianfeng/Documents/GitHub/Motor-Imagery-BCI/new_data/S001R03.edf"
        # raw = mne.io.read_raw_edf(file)

        pass
            

    def labeling(self):
        pass


