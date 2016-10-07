#!/usr/bin/python

## exploring tonotopic maps using music, cqt feature
## tonotopy_range (29 bands); one session of music stimuli

## single alpha; 10 bootstraps; no nonlinearity applied before
## temporal downsampling; just auditory cortex; 4 delays
## no zscoring stimuli, just centering; trimmed first and last 5 TRs

## 20150608AN experiment; TODO: update so don't need to change all
## experiments every single time.....

import cortex
import numpy as np
import sys
import os
from glabtools import io
from music.utils import interpdata
from music.utils import stimulus_utils_fi
from collections import defaultdict

# regression --> regression_code.huth for this to work
# utils.util --> util for this to work
from regression_code.huth.npp import mcorr
from regression_code.huth.utils import zscore, make_delayed, center
from regression_code.huth.ridge import bootstrap_ridge
from music.utils.response_utils_fi import load_response_imagedocs_music_AN
from music.utils.response_utils_fi import load_responses
from music.utils.stimulus_utils_fi import load_trfiles
from music.utils.response_utils_fi import selectROI
from music.utils.viz import separate_model_weights

# Parameters to set
endTrim = False
modelname = 'musicTon_TR_cqt_noZ_nononlin_SA'

ROIexp = '20150608AN'

subjects = ["AN"]
if len(sys.argv) > 1 and sys.argv[1] in subjects:
    subject = sys.argv[1]
    print "Got subject from command line"
else:
    subject = "AN"
    surface = subject+"fs"

trim = 5  # Trim the stimuli
nonlin = None  # nonlinearity to use when downsampling
print "Running subject {}.".format(subject)

# Open up the cloud to store stuff
inface = io.get_cc_interface()
inface.set_bucket('loganesian_stimuli') #load the damn stimuli first!

# TODO: need to update this each time depending on what extraction region
# you desire exploring; used for modelnames
## updated the cqt extraction parameters
fmin = 100 # minimum frequency extracted
num_bins = 29
bins_per_octave = 4

cqt_range = 'tonotopy_range' # TODO: update with the appropriate frequency range
stim_storage = 'music_cqt/{0}/{1}'
tpath = '/auto/k8/loganesian/projects/music/stimreports'

# Outputfiles
outdir = "/results/20150608AN"
outfile = "{0}/{1}_model_RIDGE_{2}.hdf5".format(outdir, subject, modelname)
stimoutfile = "{0}/{1}_model_{2}_stim.hf5".format(outdir, subject, modelname)
respoutfile = "{0}/{1}_model_{2}_resp.hf5".format(outdir, subject, modelname)
print outfile

# Ridge regression parameters
ROIS = ['AC']
singcutoff = 1e-10
nchunks = 20
chunklen = 16 # keeping it consistent with the tonotopy experiment
use_corr = True
single_alpha = True

if single_alpha:
    # TODO: might need to update with best alpha
    # alphas = np.array([233.572])  # From find_best_model_alpha.py
    #alphas = np.logspace(-5, 5, 20)
    alphas = np.logspace(2, 12, 20)
    nboots = 10
else:
    #alphas = np.logspace(-5, 5, 20)
    alphas = np.logspace(2, 12, 20)
    nboots = 50

# Detrending type to load
usesg = True
unwarp = True
cortexmappertype = 'trilinear'

# List of the stimuli name
# got name values by looking at SeriesDescription for images
sessions = 1 # 3 (both), 2 (20150618AN)

stimuli = ['Beethoven_Op027_No1-01', 'Beethoven_WoO080', 'Beethoven_Op031No2-01',
           'Brahms_Op010No1', 'Brahms_Op010No2', 'Chopin_Op026No1', 'Chopin_Op026No2',
           'Chopin_Op066', 'Rachmaninoff_Op036-03_Op039-01', # session 1 stimuli
#           'Bach_BWV875_0102_Beethoven_Op27No1_03','Brahms_Op005_01',
#           'Chopin_Op010_03_Op028_04_Op29', 'Chopin_Op10_04_Op48No1',
#           'Haydn_HobXVINo52_01', 'Ravel_JeuxDeau_Skyrabin_Op008No8' # session 2 stimuli 
          ]

# TRfile names
trfiles = ['20150608AN-Beethoven_Op027_No1-01.report',
           '20150608AN-Beethoven_WoO080-0.report',
           '20150608AN-Beethoven_Op031No2-01.report',
           '20150608AN-Brahms_Op010No1.report',
           '20150608AN-Brahms_Op010No2.report',
           '20150608AN-Chopin_Op026No1.report',
           '20150608AN-Chopin_Op026No2.report',
           '20150608AN-Chopin_Op066.report',
           '20150608AN-Rachmaninoff_Op036-03_Op039-01.report', # end session 1
#           '20150618AN-Bach_BWV875_0102_Beethoven_Op27No1_03.report',
#           '20150618AN-Brahms_Op005_01.report',
#           '20150618AN-Chopin_Op010_03_Op028_04_Op29.report',
#           '20150618AN-Chopin_Op10_04_Op48No1.report',
#           '20150618AN-Haydn_HobXVINo52_01.report',
#           '20150618AN-Ravel_JeuxDeau_Skyrabin_Op008No8.report' # end session 2
          ]

validation = ["Beethoven_WoO080"]

badstimuli = []  # These stimuli are ignored
Pstimuli = validation
Rstimuli = sorted(list(set(stimuli)-set(Pstimuli)-set(badstimuli)))  # Training

# Pycortex transformations
# only using session 1 transforms for now
if not unwarp:
    xfms = dict(AN="") # if unwrap need a transform
else:
    xfms = dict(AN="20150608AN")

# Load pycortex mask
mask = cortex.db.get_mask(surface, xfms[subject], cortexmappertype)

# Load response images from docdb
load_docs = dict(AN=load_response_imagedocs_music_AN)[subject]
response_images = load_docs(sessions=sessions, usesg=usesg, unwarp=unwarp)

# Remove stimulus responses that aren't in the list
for st in list(response_images.keys()):
    if st not in stimuli:
        del response_images[st]

# Load pycortex mask
mask = cortex.db.get_mask(surface, xfms[subject], cortexmappertype)

# Load responses
# Adding the line of code to only select AC
response_data = selectROI(response_images, ROIS, surface, ROIexp, mask)

Rresp = np.vstack([zscore(response_data[stimulus][5+trim:-(5+trim)].T).T
                   for stimulus in Rstimuli])
Presps = [zscore(response_data[stimulus][5+trim:-(5+trim)].T).T
          for stimulus in Pstimuli]

zRresp = Rresp
zPresps = Presps

sr = 44100
cqt_hop = 1024
seconds = 2.0
frame_length = seconds * sr
frame_length = (frame_length//cqt_hop) * cqt_hop

# Build stim_mats dictionary
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
    stim_mats[stimuli[i]] = CQTdownsamp

# Create a list of feature names and feature dimensions
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

inface.set_bucket('loganesian_tonotopy')

# Save out stimuli
inface.dict2cloud(stimoutfile, dict(zRstim=zRstim,
                                  zPstims=np.vstack(zPstims),
                                  stimlens=Rstimlens,
                                  Pstimlens=Pstimlens,
                                  modeldims=modeldims,
                                  modelnames=modelnames,
                                  delays=delays))

print "Running ridge regression.."
wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, zRresp,
                                                     delPstims[0], zPresps[0],
                                                     alphas=alphas,
                                                     nboots=nboots,
                                                     chunklen=chunklen,
                                                     nchunks=nchunks,
                                                     singcutoff=singcutoff,
                                                     single_alpha=single_alpha,
                                                     use_corr=use_corr)

print "Get total predictions for each validation stimulus.."
print "this is length of validation list {0}".format(len(delPstims))
valpreds = [np.dot(delPstim, wt) for delPstim in delPstims]
valcorr = np.vstack([mcorr(np.dot(delPstim, wt), zPresp)
                     for (delPstim, zPresp) in zip(delPstims, zPresps)])

print "Computing validation performance for sub-models.."
submodelwts = separate_model_weights(wt, ndelays, modeldims)
valmodelcorr = []

for delPstim, zPresp in zip(delPstims, zPresps):
    sepdelPstim = separate_model_weights(delPstim.T, ndelays, modeldims)
    valmodelcorr.append(np.vstack([mcorr(np.dot(s.T, w), zPresp)
                                   for (s, w) in
                                   zip(sepdelPstim, submodelwts)]))

print "Saving results.."
modelcorr = np.nan_to_num(np.array(valmodelcorr))
inface.dict2cloud(outfile, dict(models=modelnames,
                              wt=wt,
                              corr=corr,
                              valcorr=np.nan_to_num(valcorr),
                              alphas=alphas,
                              bscorrs=bscorrs,
                              valinds=valinds,
                              valmodelcorr=modelcorr,
                              mask=mask
                              )) 
