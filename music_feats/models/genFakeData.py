## this was a script used to generate fake data.
## I think I used mark's generate fake data code as inspiration for doing this
## generates boldlike responses uses the feature matrices of the actual training data
## and puts the activity in V1 (where we know we have very minimal EV/prediction performance
## otherwise)

## see documentation in music.utils.generateTestResp for full documentation

## you need to use this script to generate fake data first before fitting a model
## using a different script.
## see /auto/k8/loganesian/projects/music/tmp for examples on how to name test data

import music.utils.generateTestResp as genTest
from regression_code.huth.utils import make_delayed
from util import load_obj

exp = '20150608AN'
method='boldlike'
fname='boldlike_stdnoise1.0'
fpath='/auto/k7/lucine/projects/music/tmp'
validation = "Beethoven_WoO080"
### SCRAMBLE PARAMETERS ###
chunklen=1
action='DetrendSGolay'
valreps = 2
### BOLDLIKE PARAMETERS ###
rois = ['V1']
valreps = 2
subj = 'ANfs'
xfms = '20150608AN'
rho = 0.8
needft = True
wts=None
featurematrices = []
std = 1.0
ave = 0.0
norm = True

### Features ###
stimuli = [
           'Beethoven_Op027_No1-01', 'Beethoven_WoO080', 'Beethoven_Op031No2-01',
           'Brahms_Op010No1', 'Brahms_Op010No2', 'Chopin_Op026No1', 'Chopin_Op026No2',
           'Chopin_Op066', 'Rachmaninoff_Op036-03_Op039-01'
          ]
# to keep track of which stimulus to use for validation
valind = stimuli.index(validation)
ndelays = 4
delays = range(1, ndelays+1)
featsets = ['RMS',
            'ZCR',
            'spectralCentroid',
            'spectralSpread',
            'spectralFlatness',
            'spectralContrast',
            'spectralRolloff',
            'MFCC',
            'logMELSPECT',
            'logS',
            'chroma',
            'tonalCentroid']

# Create feature matrices to pass into function
if needft:
	for i in range(len(stimuli)):
	    curr_stim = load_obj('music', stimuli[i])
	    time_len = curr_stim.returnFeature('RMS').shape[1]
	    curr_stim.buildMatrix(featsets,time_len)
	    ftmat = curr_stim.downsample(time_len)
	    tmp = make_delayed(ftmat, delays)
	    featurematrices.append(tmp)

newBRs = genTest.generateTestR(exp=exp, method=method, fname=fname,
		fpath=fpath, rois=rois, featurematrices=featurematrices, valreps=valreps,
		valind=valind, subj=subj, xfms=xfms, wts=wts, rho=rho, std=std, ave=ave, norm=norm)
