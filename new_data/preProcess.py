import mne
import numpy as np
import numpy.random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

        self.raw = mne.io.read_raw_eeglab(self.file, preload=True)
        self.raw_data = self.raw.get_data()
        # get raw data info.
        self.info = self.raw.info
        # get parameters respectively.
        self.channels = self.raw.ch_names
        self.n_time_samps = self.raw.n_times
        self.time_secs = self.raw.times
        self.ch_names = self.raw.ch_names
        self.sampling_freq = self.raw.info['sfreq']
        self.epoch_raw = self.raw
        # visualize original dataset.
        # raw.plot(duration=60, start=0, scalings=dict(eeg=1e-4), n_channels=64, title='Raw signal', block=True)

    def epoch(self):
        # get events.
        event_time, event_dict = mne.events_from_annotations(self.raw)
        # get the event list without T(1)
        #print(event_time)
        target_event = [x for x in event_time if x[2] != 1]
        target_event = np.array(target_event)
        #print("target_event:", target_event.shape)

        # epoch data to range (-0.2, 0.8). Magic number to be fixed.
        self.epoch_raw = mne.Epochs(self.raw, target_event, tmin=-0.2, tmax=0.8, baseline=None)
        epoch_data = self.epoch_raw.get_data()
        #print("epoch_data:", epoch_data.shape)
        return target_event, np.array(epoch_data), event_dict
        # visualize data after epoch.
        # epoch_raw.plot(block=True, scalings=dict(eeg=1e-4))

    '''
    filter and normalize the data
    '''
    def filter(self):
        pass

    '''
    Produce the label vector with task name as labels. 
    '''
    def labeling(self, event_dict, target_event):
        task_labels = list(event_dict.keys())
        labels = []
        label_dict = {1: task_labels[0], 2: task_labels[1], 3: task_labels[2]}
        for e in target_event:
            labels.append(label_dict[e[-1]]) # e[-1]
        #labels = np.array(labels)
        #labels = labels.reshape((15, 1))
        return labels

    def one_hot_encoding(self, labels):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded


    def to_dataframe(self, data, label):
        pass

    def shuffle_and_split(self, data, label, train_percent):
        # shape = np.array(data).shape
        # data = np.array(data.reshape(shape[0], shape[1]*shape[2]))
        label = np.array(label)
        n_samples = len(data)
        shuffle_index = np.random.permutation(n_samples)
        np.random.shuffle(shuffle_index)
        data, label = data[shuffle_index], label[shuffle_index]
        training_range = int(n_samples*train_percent)
        training_data, training_label, testing_data, testing_label = \
            data[:training_range, :], label[:training_range],\
            data[training_range:, :], label[training_range:]
        return (training_data, training_label), (testing_data, testing_label)
