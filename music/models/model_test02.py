#!/usr/bin/python

## fake data test02.py
## used boldlike test data (i.e. generated new fake bold data)

import cortex
import numpy as np
import sys
from util import load_obj
from util import save_table_file
import pdb

# regression --> regression_code.huth for this to work
# utils.util --> util for this to work
from regression_code.huth.npp import mcorr
from regression_code.huth.utils import zscore, make_delayed
from regression_code.huth.autoreg import make_xnl_arp_x
from regression_code.huth.ridge import bootstrap_ridge
from music.utils.response_utils_fi import load_response_imagedocs_test
from music.utils.response_utils_fi import load_responses_test
from music.utils.stimulus_utils_fi import load_trfiles
from music.utils.viz import separate_model_weights


# You might not need these two funcs. If not delete.
def stack_stims(modeldata, modelnames):
    return dict([(stimulus, np.hstack([modeldata[mn][stimulus]
                                       for mn in modelnames]))
                 for stimulus in stimuli])


def make_xnl_delayed(stim, ndelays):
    return np.hstack([make_xnl_arp_x(st, ndelays, 2).T for st in stim])

# Parameters to set
modelname = "test02"
subjects = ["TEST02"]
if len(sys.argv) > 1 and sys.argv[1] in subjects:
    subject = sys.argv[1]
    print "Got subject from command line"
else:
    subject = "TEST02"
    surface = "ANfs"
    mask_to_use = 'AN' # to disambiguate subject (i.e. test) from the original (i.e. 'AN')
trim = 5  # Trim the stimuli
print "Running subject {}.".format(subject)

# Outputfiles
reading_dir = '/auto/k7/lucine/projects/music/tmp' # fake responses
outdir = "/auto/k7/lucine/projects/music/results/TEST" # test response model
outfile = "{0}/{1}_model_{2}.hdf5".format(outdir, subject, modelname)
stimoutfile = "{0}/{1}_model_{2}_stim.hf5".format(outdir, subject, modelname)
respoutfile = "{0}/{1}_model_{2}_resp.hf5".format(outdir, subject, modelname)
print outfile

# Ridge regression parameters
singcutoff=1e-10
nchunks=20
chunklen = 40
use_corr = True
single_alpha = False
if single_alpha:
    # alphas = np.array([233.572])  # From find_best_model_alpha.py
    alphas = np.array([183.298])  # From find_best_model_alpha.py
    # alphas = np.array([379.269])
    nboots = 1
else:
    alphas = np.logspace(-5, 5, 20)
    nboots = 10

# Detrending type to load
usesg = True
unwarp = True
cortexmappertype = 'trilinear'

# List of the stimuli name
# got name values by looking at SeriesDescription for images
teststim = [ 
             "20150608AN_testR_boldlike_ROI['V1']_iter0",
             "20150608AN_testR_boldlike_ROI['V1']_iter1",
             "20150608AN_testR_boldlike_ROI['V1']_iter2",
             "20150608AN_testR_boldlike_ROI['V1']_iter3",
             "20150608AN_testR_boldlike_ROI['V1']_iter4",
             "20150608AN_testR_boldlike_ROI['V1']_iter5",
             "20150608AN_testR_boldlike_ROI['V1']_iter6",
             "20150608AN_testR_boldlike_ROI['V1']_iter7",
             "20150608AN_testR_boldlike_ROI['V1']_val0",
             "20150608AN_testR_boldlike_ROI['V1']_val1"
           ]

stimuli = [
           'Beethoven_Op027_No1-01', 'Beethoven_WoO080', 'Beethoven_Op031No2-01',
           'Brahms_Op010No1', 'Brahms_Op010No2', 'Chopin_Op026No1', 'Chopin_Op026No2',
           'Chopin_Op066', 'Rachmaninoff_Op036-03_Op039-01'
          ]

validation = "Beethoven_WoO080"
# need this to account for the naming convention of test data
valNames = [
            "20150608AN_testR_boldlike_ROI['V1']_val0",
            "20150608AN_testR_boldlike_ROI['V1']_val1"
           ]
badstimuli = []  # These stimuli are ignored
Pstimuli = [validation]  # Validation
Rstimuli = sorted(list(set(stimuli)-set(Pstimuli)-set(badstimuli)))  # Training for stim
Tstimuli = sorted(list(set(teststim)-set(valNames)-set(badstimuli))) # Training for fake

# Pycortex transformations
# only using session 1 transforms for now
if not unwarp:
    xfms = dict(AN="") # if unwrap need a transform
else:
    xfms = dict(AN="20150608AN")

# Load response images from disc
response_images = load_response_imagedocs_test(reading_dir, teststim, validation=validation)

# Remove stimulus responses that aren't in the list
# NOTE: not necessary because the stimuli don't really matter for test data
# TODO: what is the best way to deal with this
# for st in list(response_images.keys()):
#     if st not in stimuli:
#         del response_images[st]

# Load pycortex mask
mask = cortex.db.get_mask(surface, xfms[mask_to_use], cortexmappertype)

# Load responses
response_data = load_responses_test(response_images, mask, force_reload=False)
# Rresp = np.vstack([zscore(response_data[stimulus][5+trim:-(5+trim)].T).T
#                    for stimulus in Tstimuli])
# Presps = [zscore(response_data[stimulus][5+trim:-(5+trim)].T).T
#           for stimulus in Pstimuli]
# TODO: how much to trim by?
Rresp = np.vstack([zscore(response_data[stimulus][trim:-trim].T).T
                   for stimulus in Tstimuli])
Presps = [zscore(response_data[stimulus][trim:-trim].T).T
          for stimulus in Pstimuli]
zRresp = Rresp
zPresps = Presps

# Load motion correction parameters
# response_mcparams = load_mcparams(response_images)
# Rmcparams = np.vstack([response_mcparams[stimulus][5+trim:-(5+trim)]
#                        for stimulus in Rstimuli])

# The selection of features
featsets = ['RMS',
            'ZCR',
            'spectralCentroid',
            'spectralSpread',
            'spectralFlatness',
            'spectralContrast',
            'spectralRolloff',
            'MFCC',
            'MELSPECT',
            'logS',
            'chroma',
            'tonalCentroid',
            'keyTimeSeries',
            'modeTimeSeries']

# Build stim_mats dictionary
stim_mats = dict()
for i in range(len(stimuli)):
    curr_stim = load_obj('music', stimuli[i])
    time_len = curr_stim.returnFeature('RMS').shape[1]
    curr_stim.buildMatrix(featsets,time_len)
    stim_mats[stimuli[i]] = curr_stim.downsample(time_len)

# Create a list of feature names and feature dimensions
modeldims = []
modelnames = featsets
for f in featsets:
    modeldims.append(curr_stim.returnFeature(f).shape[0])

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
