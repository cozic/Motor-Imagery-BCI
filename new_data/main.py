import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from preProcess import preProcess
import matplotlib.pyplot as plt
from RF import RF
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    '''
    this is for reading .set files. 
    '''
    # sub_file = '\\UCSD\\2023 winter semester\\COGS 189\\final project\\sub-001\\eeg\\sub-001_task-motion_run-3_eeg.set'
    # sub_data = mne.io.read_raw_eeglab(sub_file, preload=True)
    # raw_sub_data = sub_data.get_data()
    # sub_info = sub_data.info
    # print(sub_data['Fc5'][1])
    # print(sub_info)
    # # start, stop = sub_data.time_as_index([0.1, 0.15])
    # # data, times = sub_data[:3, start:stop]
    # # print(data.shape)
    # # print(times.shape)
    # data, times = sub_data[:, :]
    # print(raw_sub_data.shape)
    # print(data.shape, times.shape)
    # print(times.max())

    # Read the edf file.
    # file = "\\UCSD\\2023 winter semester\\COGS 189\\final project\\sourcedata\\rawdata\\S001\\S001R03.edf"
    # raw = mne.io.read_raw_edf(file)
    # raw_data = raw.get_data()
    # # get raw data info.
    # info = raw.info
    # # get parameters respectively.
    # channels = raw.ch_names
    # n_time_samps = raw.n_times
    # time_secs = raw.times
    # ch_names = raw.ch_names
    # sampling_freq = raw.info['sfreq']
    # # visualize original dataset.
    # # raw.plot(duration=60, start=0, scalings=dict(eeg=1e-4), n_channels=64, title='Raw signal', block=True)
    # # get events.
    # event_time, event_dict = mne.events_from_annotations(raw)
    # # get the event list without T(1)
    # target_event = np.array([x for x in event_time if x[2] != 1])
    # print(target_event)
    #
    # # to be filtered
    #
    # # epoch data to range (-0.2, 0.8)
    # epoch_raw = mne.Epochs(raw, target_event, tmin=-0.2, tmax=0.8, baseline=None)
    # epoch_data = epoch_raw.get_data()
    # print(epoch_data.shape)
    # print(target_event.shape)
    # # visualize data after epoch.
    # #epoch_raw.plot(block=True, scalings=dict(eeg=1e-4))


    '''
    test
    '''
    #file = 'D:\\UCSD\\2023 winter semester\\COGS 189\\Motor-Imagery-BCI\\new_data\\new_dataset\\sub-001\\eeg\\sub-001_task-motion_run-3_eeg.set'

    # check if the path exists.
    participants_info = pd.read_csv('participant.tsv', sep='\t')
    participants = participants_info.participant_id.tolist()
    files = []
    paths = []
    for sub in participants:
        #new_data / new_dataset / sub - 001 / eeg
        sub_path = 'new_dataset\\' + sub + '\\eeg\\'
        part_files = [fn for fn in os.listdir(sub_path) if fn.endswith('set')
                      & (fn.find('task-motion_run-2') == -1) & (fn.find('task-motion_run-1') == -1)]
        files.append(part_files)
        paths.append(sub_path)
    total_data = []
    total_label = []
    step = 0
    for i in range(len(files)):
        for file in files[i]:
            path = 'new_dataset\\' + participants[i] + '\\eeg\\'
            EEG = preProcess(path+file)
            # # filter the raw data with FFT and bandpass filter.
            # # bandwidth is Mu(8-12hz) and Beta(12-30hz).
            print(EEG.raw_data.shape)
            EEG.raw_data = EEG.filter(12, 30, EEG.raw_data)
            epoch, target_event, event_dict = EEG.epoch()
            # # construct CSP filter
            # CSP = CSP(epoch_data, target_event)

            #filtered_data = EEG.filter(8, 30, epoch)
            if step == 0:
                total_data = epoch
            else:
                total_data = mne.concatenate_epochs([total_data, epoch])
            label = EEG.labeling(event_dict, target_event)
    #         shape = np.array(epoch_data).shape
    #         epoch_data = epoch_data.reshape(shape[0], shape[1]*shape[2])
    #         epoch_data = epoch_data.reshape(epoch_data.shape[1], epoch_data.shape[0])
    #         # concatenate data altogether
            if total_label == []:
    #             total_data = epoch_data
                total_label = label
            else:
    #           total_data = np.concatenate((total_data, epoch_data), axis=1)
                total_label = total_label+label
            #print(f"event:{epoch.events[:, -1]}")
            step = 1
    # total_data = np.rot90(total_data)
    data = total_data.get_data()
    total_label = list(total_label)
    unique_event = list(set(total_label))
    total_event_dict = {event_name: i+1 for i, event_name in enumerate(unique_event)}
    label = [total_event_dict[event] for event in total_label]
    label = np.array(label)
    print(f"total label shape:{label}")
    #label = total_data.events[:, -1]
    csp = CSP(n_components = 4, reg=None, log=True, norm_trace=False)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf = Pipeline([('CSP', csp), ('RF', rf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = []
    for train_idx, test_idx in cv.split(data, label):
        data_train, data_test = data[train_idx], data[test_idx]
        label_train, label_test = label[train_idx], label[test_idx]

        clf.fit(data_train, label_train)
        prediction = clf.predict(data_test)
        accuracy.append(accuracy_score(label_test, prediction))
    mean_accuracy = np.mean(accuracy)
    print(f"label size:{len(set(total_label))}")
    print(f"mean accuracy: {mean_accuracy:.4f}")
    #print(np.array(total_label).shape)
    #print(np.array(total_data).shape)
    # concatenate epoch data with labels
    #print(label.shape, epoch_data.shape)
    # one hot encoding
    #onehot_encoding = EEG.one_hot_encoding(label)
    #print(onehot_encoding)
    #  shuffle and training_testing dataset.
    # training, testing = EEG.shuffle_and_split(total_data, total_label, 0.8)
    # model = RF(training, testing)
    # model.train()
    # model.test()
    # print(model.loss_and_accuracy())

