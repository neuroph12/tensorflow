import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget
import mne
from mne.io import concatenate_raws
import torch
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader


def data_loader(x, t, batch_size, shuffle=False, gpu=False, pin_memory=True):
  x = torch.from_numpy(x).float()
  t = torch.from_numpy(t).long()

  if gpu:
    x = x.cuda()
    t = t.cuda()

  set = TensorDataset(x, t)
  loader = DataLoader(set, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory)
  return loader

def split_data(X, y, split_rate=0.8):
    X, y = shuffle(X, y)
    X_train, y_train = X[:int(X.shape[0]*split_rate)], y[:int(y.shape[0]*split_rate)]
    X_val, y_val = X[int(X.shape[0]*split_rate):], y[int(y.shape[0]*split_rate):]
    return X_train, y_train, X_val, y_val

def get_data_multi(sub_id_range=[1, 50], event_code=[5,6,9,10,13,14], t=[1, 4.1], filter=[0.5,36]):
    physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,event_code) for sub_id in range(sub_id_range[0],sub_id_range[1])]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths]
    raw = concatenate_raws(parts)
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=t[0], tmax=t[1], proj=False, picks=picks,
                    baseline=None, preload=True)
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
    return X, y

def get_data_one_class_multi(sub_id_range=[1, 50], event_code=[5,6,9,10,13,14], t=[1, 4.1], filter=[0.5,36], classid=2):
    physionet_paths = [mne.datasets.eegbci.load_data(sub_id,event_code) for sub_id in range(sub_id_range[0],sub_id_range[1])]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths]
    raw = concatenate_raws(parts)
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                    exclude='bads')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    
    epoched = mne.Epochs(raw, events, classid, tmin=t[0], tmax=t[1], proj=False, picks=eeg_channel_inds,
                    baseline=None, preload=True)
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
    return X, y

def get_crops_multi(sub_id_range=[1, 50], event_code=[5,6,9,10,13,14], t=[0, 4.0], filter=[0.5,36],
                    time_window=1.0, time_step=0.5):
    physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,event_code) for sub_id in range(sub_id_range[0],sub_id_range[1])]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths]

    raw = concatenate_raws(parts)
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=t[0], tmax=t[1], proj=False, picks=picks,
                    baseline=None, preload=True)

    ### startからendまでcrop,初期配列
    start = t[0]
    end = start + time_window
    this_epoch = epochs.copy().crop(tmin=start, tmax=end)
    x = (this_epoch.get_data()*1e6).astype(np.float32)
    y = (this_epoch.events[:,2]-2).astype(np.int64)  
    print('get_time {} to {}'.format(start, end))

    ### 繰り返しcropデータを連結
    while True:
        start += time_step
        end = start + time_window
        if end > t[1]:
            break
        this_epoch = epochs.copy().crop(tmin=start, tmax=end)
        x = np.vstack((x, (this_epoch.get_data()*1e6).astype(np.float32)))
        y = np.hstack((y, (this_epoch.events[:,2]-2).astype(np.int64)))   
        print('get_time {} to {}'.format(start, end))

    return x, y

def get_crops_multi_one_class(sub_id_range=[1, 50], event_code=[5,6,9,10,13,14], t=[0, 4.0], filter=[0.5,36],
                              time_window=1.0, time_step=0.5, classid=2):
    physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,event_code) for sub_id in range(sub_id_range[0],sub_id_range[1])]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths]

    raw = concatenate_raws(parts)
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                    exclude='bads')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = mne.Epochs(raw, events, classid, tmin=t[0], tmax=t[1], proj=False, picks=eeg_channel_inds,
                    baseline=None, preload=True)

    ### startからendまでcrop,初期配列
    start = t[0]
    end = start + time_window
    this_epoch = epochs.copy().crop(tmin=start, tmax=end)
    x = (this_epoch.get_data()*1e6).astype(np.float32)
    y = (this_epoch.events[:,2]-2).astype(np.int64)  
    print('get_time {} to {}'.format(start, end))

    ### 繰り返しcropデータを連結
    while True:
        start += time_step
        end = start + time_window
        if end > t[1]:
            break
        this_epoch = epochs.copy().crop(tmin=start, tmax=end)
        x = np.vstack((x, (this_epoch.get_data()*1e6).astype(np.float32)))
        y = np.hstack((y, (this_epoch.events[:,2]-2).astype(np.int64)))   
        print('get_time {} to {}'.format(start, end))

    return x, y

def make_class(subject_id=[1,10], problem='hf', bpfilter = [0.5, 45]):
    if problem == "hf":
        X, y = get_data_multi(sub_id_range=subject_id,
                              event_code=[6,10,14],
                              t=[0, 4.0],
                              filter=bpfilter)
  
    if problem == "lr":
        X, y = get_data_multi(sub_id_range=subject_id,
                              event_code=[4,8,12],
                              t=[0, 4.0],
                              filter=bpfilter)
  
    if problem == "lh":
        Xl, yl = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        Xh, yh = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        X = np.vstack((Xl,Xh))
        y = np.hstack((yl,yh+1))
  
    if problem == "rh":
        Xr, yr = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        Xh, yh = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        X = np.vstack((Xr,Xh))
        y = np.hstack((yr-1,yh+1))
  
    if problem == "lf":
        Xl, yl = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        Xf, yf = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        X = np.vstack((Xl,Xf))
        y = np.hstack((yl,yf))
  
    if problem == "rf":
        Xr, yr = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        Xf, yf = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        X = np.vstack((Xr,Xf))
        y = np.hstack((yr-1,yf))

    if problem == "c4":
            X1, y1 = get_data_multi(sub_id_range=subject_id, event_code=[4,8,12], filter=bpfilter, t=[0., 4])
            X2, y2 = get_data_multi(sub_id_range=subject_id, event_code=[6,10,14], filter=bpfilter, t=[0., 4])
            X = np.vstack((X1,X2))
            y = np.hstack((y1,y2+2))
    return X, y

def make_class_crop(subject_id=[1,10], problem='hf', bpfilter = [0.5, 45],
                    time_window=1.0, time_step=0.5):
 
    if problem == "hf":
        X, y = get_crops_multi(sub_id_range=subject_id,
                              event_code=[6,10,14],
                              t=[0, 4.0],
                              filter=bpfilter,
                              time_window=time_window,
                              time_step=time_step)
  
    if problem == "lr":
        X, y = get_crops_multi(sub_id_range=subject_id,
                              event_code=[4,8,12],
                              t=[0, 4.0],
                              filter=bpfilter,
                              time_window=time_window,
                              time_step=time_step)
  
    if problem == "lh":
        Xl, yl = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2,
                                          time_window=time_window,
                                          time_step=time_step)
        Xh, yh = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2,
                                          time_window=time_window,
                                          time_step=time_step)
        X = np.vstack((Xl,Xh))
        y = np.hstack((yl,yh+1))
  
    if problem == "rh":
        Xr, yr = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3,
                                          time_window=time_window,
                                          time_step=time_step)
        Xh, yh = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2,
                                          time_window=time_window,
                                          time_step=time_step)
        X = np.vstack((Xr,Xh))
        y = np.hstack((yr-1,yh+1))
  
    if problem == "lf":
        Xl, yl = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        Xf, yf = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3,
                                          time_window=time_window,
                                          time_step=time_step)

        X = np.vstack((Xl,Xf))
        y = np.hstack((yl,yf))
  
    if problem == "rf":
        Xr, yr = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        Xf, yf = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3,
                                          time_window=time_window,
                                          time_step=time_step)
        X = np.vstack((Xr,Xf))
        y = np.hstack((yr-1,yf))
    return X, y


def make_4class(subject_id=[1,10], bpfilter = [0.5, 45]):
    X1, y1 = get_data_multi(sub_id_range=subject_id,
                            event_code=[6,10,14],
                            t=[0, 4.0],
                            filter=bpfilter)
  
    X2, y2 = get_data_multi(sub_id_range=subject_id,
                            event_code=[4,8,12],
                            t=[0, 4.0],
                            filter=bpfilter)

    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2+2))
    return X, y

def make_4class_crops(subject_id=[1,10], bpfilter = [0.5, 45], time_window=1, time_step=0.5):
    X1, y1 = get_crops_multi(sub_id_range=subject_id,
                            event_code=[6,10,14],
                            t=[0, 4.0],
                            filter=bpfilter,
                            time_window=time_window,
                            time_step=time_step)
  
    X2, y2 = get_crops_multi(sub_id_range=subject_id,
                            event_code=[4,8,12],
                            t=[0, 4.0],
                            filter=bpfilter,
                            time_window=time_window,
                            time_step=time_step)

    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2+2))
    return X, y

def normal_data(sub_range=[1,10], bpfilter=[0.5, 45], problem='lr'):
    if problem == '4':
        X, y = make_4class(subject_id=sub_range,
                        bpfilter = bpfilter)
    else:
        X, y = make_class(subject_id=sub_range,
                        problem=problem,
                        bpfilter = bpfilter)
    return X, y

def crop_data(sub_range=[1,10], bpfilter=[0.5, 45], problem='lr',
              time_window=1, time_step=0.5):
    if problem == '4':
        X, y = make_4class_crops(subject_id=sub_range,
                                bpfilter = bpfilter,
                                time_window=time_window,
                                time_step=time_step)
    else:
        X, y = make_class_crop(subject_id=sub_range,
                            problem=problem,
                            bpfilter = bpfilter,
                            time_window=time_window,
                            time_step=time_step)
    return X, y

def elec_map2d(X):
    batch_size = X.shape[0]
    X_spat = np.zeros((batch_size, X.shape[-1], 1, 5, 7))
    X_spat[:, :, 0, 1, 0] = X[:, 0, :]
    X_spat[:, :, 0, 1, 1] = X[:, 1, :]
    X_spat[:, :, 0, 1, 2] = X[:, 2, :]
    X_spat[:, :, 0, 1, 3] = X[:, 3, :]
    X_spat[:, :, 0, 1, 4] = X[:, 4, :]
    X_spat[:, :, 0, 1, 5] = X[:, 5, :]
    X_spat[:, :, 0, 1, 6] = X[:, 6, :]

    X_spat[:, :, 0, 2, 0] = X[:, 7, :]
    X_spat[:, :, 0, 2, 1] = X[:, 8, :]
    X_spat[:, :, 0, 2, 2] = X[:, 9, :]
    X_spat[:, :, 0, 2, 3] = X[:, 10, :]
    X_spat[:, :, 0, 2, 4] = X[:, 11, :]
    X_spat[:, :, 0, 2, 5] = X[:, 12, :]
    X_spat[:, :, 0, 2, 6] = X[:, 13, :]

    X_spat[:, :, 0, 3, 0] = X[:, 14, :]
    X_spat[:, :, 0, 3, 1] = X[:, 15, :]
    X_spat[:, :, 0, 3, 2] = X[:, 16, :]
    X_spat[:, :, 0, 3, 3] = X[:, 17, :]
    X_spat[:, :, 0, 3, 4] = X[:, 18, :]
    X_spat[:, :, 0, 3, 5] = X[:, 19, :]
    X_spat[:, :, 0, 3, 6] = X[:, 20, :]

    X_spat[:, :, 0, 0, 0] = X[:, 30, :]
    X_spat[:, :, 0, 0, 1] = X[:, 31, :]
    X_spat[:, :, 0, 0, 2] = X[:, 32, :]
    X_spat[:, :, 0, 0, 3] = X[:, 33, :]
    X_spat[:, :, 0, 0, 4] = X[:, 34, :]
    X_spat[:, :, 0, 0, 5] = X[:, 35, :]
    X_spat[:, :, 0, 0, 6] = X[:, 36, :]

    X_spat[:, :, 0, 4, 0] = X[:, 47, :]
    X_spat[:, :, 0, 4, 1] = X[:, 48, :]
    X_spat[:, :, 0, 4, 2] = X[:, 49, :]
    X_spat[:, :, 0, 4, 3] = X[:, 50, :]
    X_spat[:, :, 0, 4, 4] = X[:, 51, :]
    X_spat[:, :, 0, 4, 5] = X[:, 52, :]
    X_spat[:, :, 0, 4, 6] = X[:, 53, :]
    return X_spat

def elec_map2d_full(X):
    batch_size = X.shape[0]
    X_spat = np.zeros((batch_size, X.shape[-1], 1, 11, 11))
    X_spat[:, :, 0, 4, 2] = X[:, 0, :]
    X_spat[:, :, 0, 4, 3] = X[:, 1, :]
    X_spat[:, :, 0, 4, 4] = X[:, 2, :]
    X_spat[:, :, 0, 4, 5] = X[:, 3, :]
    X_spat[:, :, 0, 4, 6] = X[:, 4, :]
    X_spat[:, :, 0, 4, 7] = X[:, 5, :]
    X_spat[:, :, 0, 4, 8] = X[:, 6, :]

    X_spat[:, :, 0, 5, 2] = X[:, 7, :]
    X_spat[:, :, 0, 5, 3] = X[:, 8, :]
    X_spat[:, :, 0, 5, 4] = X[:, 9, :]
    X_spat[:, :, 0, 5, 5] = X[:, 10, :]
    X_spat[:, :, 0, 5, 6] = X[:, 11, :]
    X_spat[:, :, 0, 5, 7] = X[:, 12, :]
    X_spat[:, :, 0, 5, 8] = X[:, 13, :]

    X_spat[:, :, 0, 6, 2] = X[:, 14, :]
    X_spat[:, :, 0, 6, 3] = X[:, 15, :]
    X_spat[:, :, 0, 6, 4] = X[:, 16, :]
    X_spat[:, :, 0, 6, 5] = X[:, 17, :]
    X_spat[:, :, 0, 6, 6] = X[:, 18, :]
    X_spat[:, :, 0, 6, 7] = X[:, 19, :]
    X_spat[:, :, 0, 6, 8] = X[:, 20, :]

    X_spat[:, :, 0, 1, 3] = X[:, 21, :]
    X_spat[:, :, 0, 1, 5] = X[:, 22, :]
    X_spat[:, :, 0, 1, 7] = X[:, 23, :]

    X_spat[:, :, 0, 2, 2] = X[:, 24, :]
    X_spat[:, :, 0, 2, 3] = X[:, 25, :]
    X_spat[:, :, 0, 2, 5] = X[:, 26, :]
    X_spat[:, :, 0, 2, 7] = X[:, 27, :]
    X_spat[:, :, 0, 2, 8] = X[:, 28, :]

    X_spat[:, :, 0, 3, 1] = X[:, 29, :]
    X_spat[:, :, 0, 3, 2] = X[:, 30, :]
    X_spat[:, :, 0, 3, 3] = X[:, 31, :]
    X_spat[:, :, 0, 3, 4] = X[:, 32, :]
    X_spat[:, :, 0, 3, 5] = X[:, 33, :]
    X_spat[:, :, 0, 3, 6] = X[:, 34, :]
    X_spat[:, :, 0, 3, 7] = X[:, 35, :]
    X_spat[:, :, 0, 3, 8] = X[:, 36, :]
    X_spat[:, :, 0, 3, 9] = X[:, 37, :]

    X_spat[:, :, 0, 4, 1] = X[:, 38, :]
    X_spat[:, :, 0, 4, 9] = X[:, 39, :]

    X_spat[:, :, 0, 5, 1] = X[:, 40, :]
    X_spat[:, :, 0, 5, 9] = X[:, 41, :]
    X_spat[:, :, 0, 5, 0] = X[:, 42, :]
    X_spat[:, :, 0, 5, 10] = X[:, 43, :]

    X_spat[:, :, 0, 6, 1] = X[:, 44, :]
    X_spat[:, :, 0, 6, 9] = X[:, 45, :]

    X_spat[:, :, 0, 7, 1] = X[:, 46, :]
    X_spat[:, :, 0, 7, 2] = X[:, 47, :]
    X_spat[:, :, 0, 7, 3] = X[:, 48, :]
    X_spat[:, :, 0, 7, 4] = X[:, 49, :]
    X_spat[:, :, 0, 7, 5] = X[:, 50, :]
    X_spat[:, :, 0, 7, 6] = X[:, 51, :]
    X_spat[:, :, 0, 7, 7] = X[:, 52, :]
    X_spat[:, :, 0, 7, 8] = X[:, 53, :]
    X_spat[:, :, 0, 7, 9] = X[:, 54, :]
    
    X_spat[:, :, 0, 8, 2] = X[:, 55, :]
    X_spat[:, :, 0, 8, 3] = X[:, 56, :]
    X_spat[:, :, 0, 8, 5] = X[:, 57, :]
    X_spat[:, :, 0, 8, 7] = X[:, 58, :]
    X_spat[:, :, 0, 8, 8] = X[:, 59, :]

    X_spat[:, :, 0, 9, 3] = X[:, 60, :]
    X_spat[:, :, 0, 9, 5] = X[:, 61, :]
    X_spat[:, :, 0, 9, 7] = X[:, 62, :]

    X_spat[:, :, 0, 10, 5] = X[:, 63, :]

    return X_spat
