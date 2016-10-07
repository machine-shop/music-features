## script for extracting cqt feature from the speech session stimuli
## saves features to the cloud under a directory name with the parameters
## with which the features were extracted

import librosa
import numpy as np
import os
from glabtools import io
from scipy import signal

## script for extracting the cqt feature
## different ranges saved in different directories

inface = io.get_cc_interface()
inface.set_bucket('loganesian_stimuli')

### STIMULUS INFORMATION. TO BE FORMATTED LATER ###

stimuli = ['alternateithicatom',
            'wheretheressmoke', # validation
            'avatar',
            'legacy',
            'odetostepfather',
            'souls',
          ]

wavnames = ['alternateithicatom',
            'wheretheressmoke-norm-1', # validation
            'avatar-norm',
            'legacy',
            'odetostepfather',
            'souls',
           ]

stim_out = 'speech_cqt/{0}/{1}'
extractionDescription = '36bins_6binsperoctave_100hzmin'

# multiply the frames with a hanning window; default: False
# librosa already multiplies i think
use_han = False

sr = 44100 # sampling rate of the wav files

fpath = '/auto/k2/stimuli/audio/wavs'
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
                                  fmin=fmin, hop_length=cqt_hop))
        CQT_frames.append(CQTf[:,1:-1])
    # concatenate everything together
    CQT = np.hstack(CQT_frames)
    
    # Take the log amplitude
    CQTlog = librosa.logamplitude(CQT**2, ref_power=np.max)

    # save the extracted CQT
    inface.upload_raw_array(stim_out.format(extractionDescription, stimuli[i]),
                            CQTlog)
