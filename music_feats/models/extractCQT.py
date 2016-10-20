## script for extracting constant q transform feature from music; session 1 and 2 done separately
## extracted features are saved to the could

## there are several different hearing ranges that were extracted; the ranges are all available below
## pr (piano range), tr (29 band tonotopy localizer range), tr2 (128 band tonotopy localier range),
## hhr (human hearing range), fr (formisano 2012 range), fr2 (modified formisano range, differ in binning)

import librosa
import numpy as np
import os
from glabtools import io
from scipy import signal

## script for extracting the cqt feature
## different ranges saved in different directories

## current tonotopy range: num_bins: 29, bins per octave: 4, fmin: 100hz


inface = io.get_cc_interface()
inface.set_bucket('loganesian_stimuli')

session = 1

if session == 1:
    stimuli = ['Beethoven_Op027_No1-01', 'Beethoven_WoO080', 'Beethoven_Op031No2-01',
               'Brahms_Op010No1', 'Brahms_Op010No2', 'Chopin_Op026No1', 'Chopin_Op026No2',
               'Chopin_Op066','Rachmaninoff_Op036-03_Op039-01',]

    wavnames = ['session1/Beethoven_Op027No1-01_003_20090916-SMD-norm',
                'session1/Beethoven_WoO080_001_20081107-SMD-norm',
                'session1/Beethoven_Op031No2-01_002_20090916-SMD-norm',
                'session1/Brahms_Op010No1_003_20090916-SMD-norm',
                'session1/Brahms_Op010No2_003_20090916-SMD-norm',
                'session1/Chopin_Op026No1_003_20100611-SMD-norm',
                'session1/Chopin_Op026No2_005_20100611-SMD-norm',
                'session1/Chopin_Op066_006_20100611-SMD-norm',
                'session1/Rachmaninoff_Op036-03_Op039-01-SMD-norm',]

elif session == 2:
    stimuli = ['Bach_BWV875_0102_Beethoven_Op27No1_03','Brahms_Op005_01',
               'Chopin_Op010_03_Op028_04_Op29', 'Chopin_Op10_04_Op48No1',
               'Haydn_HobXVINo52_01', 'Ravel_JeuxDeau_Skyrabin_Op008No8',]

    wavnames = [ 'session2/Bach_BWV875-01-02_Beethoven_Op27-No1-03-SMD-norm',
                 'session2/Brahms_Op005-01_002_20110315-SMD-norm',
                 'session2/Chopin_Op010-03_Op028-04_Op029-SMD-norm',
                 'session2/Chopin_Op010-04_Op048No1-SMD-norm',
                 'session2/Haydn_HobXVINo52-01_008_20110315-SMD-norm',
                 'session2/Ravel_JeuxDEau_Skyrabin_Op008No8-SMD-norm']

stim_out = 'music_cqt/{0}/{1}'

cqt_range = 'full_tonotopy_range' ## TODO: need to change this each time

pr = False # using 'piano_range'
tr = False # using 'tonotopy_range'
tr2 = True # using 'full_tonotopy_range'
hhr = False # using 'humanhearing_range'
fr = False # using 'FR_range' (FR = formisano 2012)
fr2 = False # using 'modFR2_range'

# multiply the frames with a hanning window; default: False
# librosa already multiplies i think
use_han = False

sr = 44100 # sampling rate of the wav files

fpath = '/auto/k2/stimuli/music/wavs'
wavname = '{0}.wav'

### CQT parameters ###
# 25ms hop between frames when calcualting CQT
# uncertain, but from resulting feature length, I think this gives
# 50ms resolution in window lengths; trying to keep consistent with
# other features and with librosa default
cqt_hop = 1024
seconds = 2.0 # chunk size prior to CQT
frame_length = seconds * sr # convert seconds to samples
# make frame length divisible by the hop length of cqt to
# keep things consistent
frame_length = (frame_length//cqt_hop) * cqt_hop 
frame_hop = frame_length - cqt_hop # so that we can splice and concat

if pr:
    n_bins = 30 # span 7.5 ocatves, up to 5kHz
    bins_per_octave = 4
    fmin = 27.5 # lowest frequency on a piano
elif tr:
    n_bins = 29
    bins_per_octave = 4
    fmin = 100
elif tr2:
    n_bins = 128
    bins_per_octave = 18
    fmin = 100
elif hhr:
    n_bins = 120
    bins_per_octave = 12
    fmin = 20
elif fr:
    n_bins = 36 #31
    bins_per_octave = 7 #6
    fmin = 186
elif fr2:
    n_bins = 40
    bins_per_octave = 8
    fmin = 186
else:
    n_bins = 36
    bins_per_octave = 6
    fmin = 100

for i in range(len(stimuli)):
    # load file
    wavpath = os.path.join(fpath, wavname.format(wavnames[i]))
    y, rate = librosa.load(wavpath, sr=sr)

    # pad the signal so librosa frame doesn't cut things off
    padded_y = np.append(y, np.zeros(frame_length))
    
    # chunk the initial signal
    y_frames = librosa.util.frame(padded_y, frame_length=frame_length,
                                  hop_length=frame_hop)

    # hanning window to smooth the spectrum out
    han_win = signal.hanning(frame_length)

    # let's extract
    CQT_frames = []
    for frame in range(y_frames.shape[1]):
        if not use_han:
            sig = y_frames[:, frame]
        else:
            sig = han_win * y_frames[:, frame]

        CQTf = np.abs(librosa.cqt(sig, sr=sr, n_bins=n_bins,
                                  bins_per_octave=bins_per_octave,
                                  fmin=fmin, hop_length=cqt_hop,
                                  real=False))
        CQT_frames.append(CQTf[:,1:-1])
    # concatenate everything together
    CQT = np.hstack(CQT_frames)
    
    # Take the log amplitude
    CQTlog = librosa.logamplitude(CQT**2, ref_power=np.max)

    # save the extracted CQT
    inface.upload_raw_array(stim_out.format(cqt_range, stimuli[i]),
                            CQTlog)
