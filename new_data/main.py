import mne
import numpy as np
import preProcess
import matplotlib.pyplot as plt
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
    file = "/Users/lijianfeng/Documents/GitHub/Motor-Imagery-BCI/new_data/S001R03.edf"
    raw = mne.io.read_raw_edf(file,stim_channel='Event marker', misc=['Temp rectal'])
    raw.load_data() 
    raw_data = raw.get_data()
    # get raw data info.
    info = raw.info
    # get parameters respectively.
    channels = raw.ch_names
    n_time_samps = raw.n_times
    time_secs = raw.times
    ch_names = raw.ch_names
    sampling_freq = raw.info['sfreq']
    # visualize original dataset.
    # raw.plot(duration=60, start=0, scalings=dict(eeg=1e-4), n_channels=64, title='Raw signal', block=True)
    # get events.
    event_time, event_dict = mne.events_from_annotations(raw)
    # get the event list without T(1)
    target_event = np.array([x for x in event_time if x[2] != 1])
    print(target_event)

    # to be filtered

    # epoch data to range (-0.2, 0.8)
    epoch_raw = mne.Epochs(raw, target_event, tmin=-0.2, tmax=0.8, baseline=None)
    epoch_data = epoch_raw.get_data()
    print(epoch_data.shape)
    print(target_event.shape)
    # visualize data after epoch.
    
    #raw = mne.io.read_raw_edf(f, stim_channel=‘Event marker’, misc=[‘Temp rectal’])
    #raw.load_data()
    #epoch_raw.filter()

    low_cut = 0.1
    hi_cut  = 30
    raw_filt = raw.copy().filter(low_cut, hi_cut)

    epoch_raw.plot(block=True, scalings=dict(eeg=1e-4))


    # '''
    # actual main
    # '''
    # # check if the path exists.
    # file = "\\UCSD\\2023 winter semester\\COGS 189\\final project\\sourcedata\\rawdata\\S001\\S001R03.edf"
    # EEG = preProcess(file)

