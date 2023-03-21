import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_eeglab
from mne.decoding import CSP

class CSP:
    def __init__(self, epoched_data, all_events):
        self.epoch = epoched_data
        self.events = all_events
        # initialize the CSP_filters with shape(num_events, num_channel)
        # self.CSP_filters = np.zeros((len(self.events), 64))
        self.CSP_filters = []
    '''
    Compute the covariance matrix of each epoch in the individual classes 
    and average the cov_matrix. 
    @param: epoched_data: the data being epoched
            all_event: set of unique events in the epoched_data(label) 
    '''
    def cov_matrix_and_average(self):
        for event in self.events:
            one_class_epoch, rest_class_epoch = self.separate_into_classes(event)
            # compute the cov_matrix for one class
            cov_matrix = [np.dot(epoch, epoch.T)/np.trace(np.dot(epoch, epoch.T))
                          for _, epoch in one_class_epoch]



            # compute the cov_matrix for rest class
            for epoch in rest_class_epoch:
                pass

        # dimensionality reduction for the two feature matrices.
        pass

    '''
    separate the epoched_data into one and rest, and compute their 
    covariance matrix accordingly. 
    @param: event: the individual event(class)
    '''
    def separate_into_classes(self, event):


        return single_class_epoch, rest_class_epoch
