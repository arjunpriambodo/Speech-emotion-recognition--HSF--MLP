import glob
import os
import librosa
import numpy as np
import pickle

data_path = 'data/speech'
files = glob.glob(os.path.join(data_path + '/*/', '*.wav'))
files.sort()

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, sr=None)
    stft = np.abs(librosa.stft(X))
    mfcc = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    mfcc_std = np.std(librosa.feature.mfcc(
        y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
    return(mfcc,
           mfcc_std)

# membuat daftar kosong penyimpanan fitur dan label
feat = [] # spectral feature
lab = [] # label emosi

# iterasi semua file
for file in files:
    print("Extracting features from ", file)
    feat_i = np.hstack(extract_feature(file))
    lab_i = os.path.basename(file).split('-')[2]
    feat.append(feat_i)
    lab.append(int(lab_i)-1) # label dimulai dari 0

np.save(data_path + 'x.npy', feat)
np.save(data_path + 'y.npy', lab)