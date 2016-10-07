## script used to load the cqt features for the music stimuli
## from session 1; all of the design matrices are centered but not
## zscored

## final output: delRstim and delPstims (which are delayed versions of
## training and validation matrices)
## 4 delays; no nonlinearity applied before downsampling; trims first
## and last 5 TRs

## can modify which range of frequencies (for the cqt) to use for the
## feature matrix

from glabtools import io
import sys
import numpy as np
import os
from music.utils import interpdata
from music.utils import stimulus_utils_fi
from collections import defaultdict

from regression_code.huth.utils import zscore, make_delayed, center

trim = 5
nonlin = None
endTrim = False

inface = io.get_cc_interface('loganesian_stimuli')

cqt_range = 'modFR2_range'
fmin =186
num_bins = 40
bins_per_octave = 8

stim_storage = 'music_cqt/{0}/{1}'
tpath = '/auto/k8/loganesian/projects/music/stimreports'

stimuli = ['Beethoven_Op027_No1-01', 'Beethoven_WoO080', 'Beethoven_Op031No2-01',
           'Brahms_Op010No1', 'Brahms_Op010No2', 'Chopin_Op026No1', 'Chopin_Op026No2',
           'Chopin_Op066', 'Rachmaninoff_Op036-03_Op039-01']

trfiles = ['20150608AN-Beethoven_Op027_No1-01.report',
           '20150608AN-Beethoven_WoO080-0.report',
           '20150608AN-Beethoven_Op031No2-01.report',
           '20150608AN-Brahms_Op010No1.report',
           '20150608AN-Brahms_Op010No2.report',
           '20150608AN-Chopin_Op026No1.report',
           '20150608AN-Chopin_Op026No2.report',
           '20150608AN-Chopin_Op066.report',
           '20150608AN-Rachmaninoff_Op036-03_Op039-01.report']

validation = ['Beethoven_WoO080']

badstimuli = []  # These stimuli are ignored
Pstimuli = validation
Rstimuli = sorted(list(set(stimuli)-set(Pstimuli)-set(badstimuli)))  # Training


sr = 44100
cqt_hop = 1024
seconds = 2.0
frame_length = seconds * sr
frame_length = (frame_length//cqt_hop) * cqt_hop

# Build stim_mats dictionary
stim_mats_nonDS = dict()
stim_mats = dict()
for i in range(len(stimuli)):
    # load the cqt for the stimulus
    CQT = inface.download_raw_array(stim_storage.format(cqt_range, stimuli[i]))

    # calculate the new times (after downsampling)
    tr = stimulus_utils_fi.TRFile(os.path.join(tpath, trfiles[i]))

    newtimes = tr.get_reltriggertimes()
    if endTrim:
        newtimes = newtimes[trim:-trim]
    else:
        newtimes = newtimes[trim:]

    # calculate the old times of the cqt data
    num_frames = CQT.shape[1]/(frame_length/cqt_hop - 1)
    sig_len = num_frames * frame_length - (num_frames - 1) * cqt_hop
    time_per_cqt_samp = (sig_len / CQT.shape[1])/sr

    oldtimes = np.arange(0.0, sig_len/sr, time_per_cqt_samp)

    if len(oldtimes) > CQT.shape[1]:
        oldtimes = oldtimes[:CQT.shape[1]]
    elif len(oldtimes) < CQT.shape[1]:
        oldtimes = np.append(oldtimes, np.asarray(oldtimes[-1]+time_per_cqt_samp))

    # downsample cqt mat
    CQTdownsamp = interpdata.lanczosinterp2D(CQT.T, oldtimes,
                                             newtimes, window=3,
                                             cutoff_mult=1.0,
                                             rectify=False)

    # build and store short duration matrix
    stim_mats_nonDS[stimuli[i]] = CQT
    stim_mats[stimuli[i]] = CQTdownsamp


start = fmin
end = start * (2**(num_bins/float(bins_per_octave)))
cqtbins = np.logspace(np.log10(start), np.log10(end), num=num_bins)

modelnames = ['{0}hz'.format(elem) for elem in cqtbins]
modeldims = [1 for i in range(len(modelnames))]


# Feature matrix
Rstim = np.vstack([center(stim_mats[stimulus][trim:-trim].T).T
                   for stimulus in Rstimuli])
Rstimlens = np.array([stim_mats[stimulus][trim:-trim].shape[0]
                      for stimulus in Rstimuli])
Pstims = [center(stim_mats[stimulus][trim:-trim].T).T
          for stimulus in Pstimuli]
Pstimlens = np.array([stim_mats[stimulus][trim:-trim].shape[0]
                      for stimulus in Pstimuli])

zRstim = Rstim
zPstims = [ps for ps in Pstims]

# Create FIR model (ndelays of the Feature Matrix)
ndelays = 4
delays = range(1, ndelays+1)
delRstim = make_delayed(zRstim, delays)
delPstims = [make_delayed(zp, delays) for zp in zPstims]

