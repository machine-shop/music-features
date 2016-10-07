#!/usr/bin/python

## fitting model on frequencies split in two
## splitting along 1600 Hz, based on norman phd
## thesis; frequency feature space will have 2
## dimensions then low/high
## only interested 200Hz-6400Hz
## fitting only on auditory cortex, single alpha (10 boots)
## zscored and 4 delays; trim first and last 5 TRs
## spectrogram feature extracted using the parameters listed in model_aa.py
## a squaring non linearity applied before temporal downsampling

import cortex
import numpy as np
import sys
from util import load_obj
from util import save_table_file
import pdb
import music.features.features as ft
from math import floor

# regression --> regression_code.huth for this to work
# utils.util --> util for this to work
from regression_code.huth.npp import mcorr
from regression_code.huth.utils import zscore, make_delayed
from regression_code.huth.autoreg import make_xnl_arp_x
from regression_code.huth.ridge import bootstrap_ridge
from music.utils.response_utils_fi import load_response_imagedocs_speech_AN
from music.utils.response_utils_fi import load_responses
from music.utils.stimulus_utils_fi import load_trfiles
from music.utils.response_utils_fi import selectROI
from music.utils.viz import separate_model_weights


# You might not need these two funcs. If not delete.
def stack_stims(modeldata, modelnames):
    return dict([(stimulus, np.hstack([modeldata[mn][stimulus]
                                       for mn in modelnames]))
                 for stimulus in stimuli])


def make_xnl_delayed(stim, ndelays):
    return np.hstack([make_xnl_arp_x(st, ndelays, 2).T for st in stim])

# Parameters to set
modelname = "al4_speech"
ROIexp = '20150608AN' #need this to extract ROI
subjects = ["AN"]
if len(sys.argv) > 1 and sys.argv[1] in subjects:
    subject = sys.argv[1]
    print "Got subject from command line"
else:
    subject = "AN"
    surface = subject+"fs"
trim = 5  # Trim the stimuli
nonlin = 'squared' # nonlinearity to use when downsampling
print "Running subject {}.".format(subject)

# Extracted feature object files
fpath = '/auto/k7/lucine/projects' # the directory the extracted features are saved in
project = 'speech' # the project directory the extracted features are saved in

# Outputfiles
outdir = "/auto/k7/lucine/projects/speech/results/20150801AN"
outfile = "{0}/{1}_model_{2}.hdf5".format(outdir, subject, modelname)
stimoutfile = "{0}/{1}_model_{2}_stim.hf5".format(outdir, subject, modelname)
respoutfile = "{0}/{1}_model_{2}_resp.hf5".format(outdir, subject, modelname)
print outfile

# Ridge regression parameters
ROIS = ['AC'] # adding this new parameter to select ROI
singcutoff=1e-10
nchunks=20
chunklen = 40
use_corr = True
single_alpha = True # single alpha; make same prior assumption for all
if single_alpha:
    # TODO: might need to update with best alpha
    # alphas = np.array([233.572])  # From find_best_model_alpha.py
    # alphas = np.array([183.298])  # From find_best_model_alpha.py
    # alphas = np.array([379.269])
    alphas = np.logspace(-5, 5, 20)
    nboots = 10
else:
    alphas = np.logspace(-5, 5, 20)
    nboots = 10

# Detrending type to load
usesg = True
unwarp = True
cortexmappertype = 'trilinear'

# List of the stimuli name
# got name values by looking at SeriesDescription for images
sessions = 1 # 1 (20150608AN), 2 (20150618AN), 3 (both)

stimuli = ['alternateithicatom',
            'wheretheressmoke', # validation
            'avatar',
            'legacy',
            'odetostepfather',
            'souls',
          ]

validation = 'wheretheressmoke'
badstimuli = []  # These stimuli are ignored
Pstimuli = [validation]  # Validation
Rstimuli = sorted(list(set(stimuli)-set(Pstimuli)-set(badstimuli)))  # Training

# Pycortex transformations
# only using session 1 transforms for now
if not unwarp:
    xfms = dict(AN="") # if unwrap need a transform
else:
    xfms = dict(AN="20150608AN")

# Load response images from docdb
load_docs = dict(AN=load_response_imagedocs_speech_AN)[subject]
response_images = load_docs(sessions=sessions, usesg=usesg, unwarp=unwarp)
# response_images[validation] = response_images[validation][:2] # take all validation

# Remove stimulus responses that aren't in the list
for st in list(response_images.keys()):
    if st not in stimuli:
        del response_images[st]

# Load pycortex mask
mask = cortex.db.get_mask(surface, xfms[subject], cortexmappertype)

# Load responses
# Adding the line of code to only select AC
response_data = selectROI(response_images, ROIS, surface, ROIexp, mask)

# original
# response_data = load_responses(response_images, mask, force_reload=False)

# back to original
Rresp = np.vstack([zscore(response_data[stimulus][5+trim:-(5+trim)].T).T
                   for stimulus in Rstimuli])
Presps = [zscore(response_data[stimulus][5+trim:-(5+trim)].T).T
          for stimulus in Pstimuli]
zRresp = Rresp
zPresps = Presps

# Load motion correction parameters
# response_mcparams = load_mcparams(response_images)
# Rmcparams = np.vstack([response_mcparams[stimulus][5+trim:-(5+trim)]
#                        for stimulus in Rstimuli])

# The selection of features
featsets = ['logS']

## calculations necessary to create proper low/high frequency groupings
sampsrate = 44100.0 # sampling rate
nfft = 2048.0 # the way the sound file was binned
deltw = sampsrate / nfft # the size of the frequency bins
lowfreqLimit = 1600.0 # 1600 Hz (or close) cutoff b/w high/low frequencies
lowfreqCount = floor(lowfreqLimit / deltw) # index for cutoff for low frequencies
lowF = 200.0 # 200 hz lower bound on frequencies
highF = 6400.0 #6400 hz upper bound on frequencies
lowFInd = floor(lowF / deltw) # index for low frequency cut off
highFInd = floor(highF / deltw) # index for high frequency cut off

# Build stim_mats dictionary
stim_mats = dict()
for i in range(len(stimuli)):
    curr_stim = load_obj(project, stimuli[i], filepath=fpath)

    # short duration features
    time_len = curr_stim.returnFeature('RMS').shape[1]
    curr_stim.buildMatrix(featsets,time_len)
    # build and store short duration matrix
    stim_mats[stimuli[i]] = curr_stim.downsampleMatrix(time_len, nonlin=nonlin)
    stim_mats[stimuli[i]] = stim_mats[stimuli[i]][:,lowFInd:highFInd+1]

# ~1600Hz will be grouped with low frequencies
# Create a list of feature names and feature dimensions
modelnames = ['lowF', 'highF']
modeldims = [lowfreqCount-(lowFInd-1), highFInd  - lowfreqCount]
# for f in featsets:
#     modeldims.append(curr_stim.returnFeatureDim(f))

# Feature matrix
Rstim = np.vstack([zscore(stim_mats[stimulus][trim:-trim].T).T
                   for stimulus in Rstimuli])
Rstimlens = np.array([stim_mats[stimulus][trim:-trim].shape[0]
                      for stimulus in Rstimuli])
Pstims = [zscore(stim_mats[stimulus][trim:-trim].T).T
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

# pdb.set_trace()

# Save out stimuli
save_table_file(stimoutfile, dict(zRstim=zRstim,
                                  zPstims=np.vstack(zPstims),
                                  # zvalPstim=zvalPstim,
                                  # zimstim=zimstim,
                                  stimlens=Rstimlens,
                                  Pstimlens=Pstimlens,
                                  # stimmeans=stimmeans,
                                  # stimstds=stimstds,
                                  modeldims=modeldims,
                                  modelnames=modelnames,
                                  # modelweights=modelweights,
                                  delays=delays))

# Save out responses
print "Saving responses.."
save_table_file(respoutfile, dict(zRresp=zRresp, zPresp=np.vstack(zPresps),
                                  mask=mask))

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
save_table_file(outfile, dict(models=featsets,
                              wt=wt,
                              corr=corr,
                              valcorr=np.nan_to_num(valcorr),
                              alphas=alphas,
                              bscorrs=bscorrs,
                              valinds=valinds,
                              valmodelcorr=modelcorr,
                              mask=mask
                              ))
