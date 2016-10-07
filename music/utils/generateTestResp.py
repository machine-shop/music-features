## all functions used to generate fake data, by either scrambling an already existing BOLD response
## or by create a new bold-like response
## see documentation of each function for details on parameters/return values

import numpy as np
import docdb
import os
import sys
import pdb
from random import shuffle, random, randint
from itertools import chain
from collections import defaultdict
from cortex import get_roi_mask
from util import save_table_file
import nibabel as ni

def generateTestR(exp, method='boldlike', fname=None,
				fpath='/auto/k7/lucine/projects/music/tmp', **kwargs):
	"""
	Generate fake data to test pipeline with. Will call a sub function to
	generate fake data in a particular way.
		:parameters:
			- exp: string. Experiment name. Used to name output files
				or to query bold responses in the scramble method.
			- method: string. Method used to generate fake data.
				:Options: 
					- 'boldlike' uses weights to create fake data from
						feature matrices. Calls boldlike(). Default.
					- 'scramble' scrambles existing bold responses.
						calls scramble()
			- fname : string. The name to save the NIFTI files as.
				Default is None.
			- fpath : string. The path where to store the resulting NIFTI files.
			- kwargs : the arguments for boldlike() or scramble(), depending on
				which method is used
	"""
	if not os.path.exists(fpath):
		os.mkdir(fpath)
	if method == 'scramble':
		# scramble an already existing BOLD response
		newBRs = scramble(exp, fname, fpath, **kwargs)
	elif method == 'boldlike':
		# create a BOLD-like response
		newBRs = boldlike(exp, fname, fpath, **kwargs)
	return newBRs

def scramble(exp, fname, fpath, chunklen=1, action='DetrendSGolay',
			validation=None, valreps=2):
	""" 
	Scramble actual BOLD responses to get fake responses.
		:parameters:
			- exp: string. Experiment name. Used to name output files
				or to query bold responses in the scramble method.
			- fname : string. The name to save the NIFTI files as.
				Default is None.
			- fpath : string. The path where to store the resulting NIFTI files.
			- chunklen : integer. The chunk size (in TRs) to scramble around.
				Default is 1 TR.
			- action : string. Which action to query the original bold responses.
				Default is 'DetrendSGolay'
			- validation : string. The stimulus/stimuli used to create the
				validation responses. Default is None.
			- valreps : integer. The number of validation responses to create.
				Default is 2 responses.

	"""
	newBRs = defaultdict(list)
	# Create the filename to save the files as
	if fname is None:
		fname = exp + '_testR_scramble_chunklen{0}'.format(chunklen)
	valname = fname + '_val{0}.nii'	
	fname = fname + '_iter{0}.nii'
	# Load the shuffled response images
	shuffIms = loadShuffledIMS(exp, action)
	cnt, vflag = 0, False
	while shuffIms:
		currIm = shuffIms.pop()
		BR = currIm.get_data()
		# Helper function that scrambles the bold responses
		BRnew = scrambleHelper(BR, chunklen)
		# make multiple validation responses
		if (currIm.SeriesDescription == validation) and (not vflag):
			vflag = True
			BRnewI = savenewBR(BRnew, fpath, valname, cnt=0)
			newBRs[validation].append(BRnewI)
			for i in range(1, valreps):
				# take first validation & add noise to it to create other vals
				tmp = addnoise(BRnew)
				BRnewI = savenewBR(tmp, fpath, valname, cnt=i)
				newBRs[validation].append(BRnewI)
		elif (currIm.SeriesDescription != validation): # training data
			BRnewI = savenewBR(BRnew, fpath, fname, cnt=cnt)
			newBRs[fname.format(cnt)].append(BRnewI)
			cnt += 1
	return newBRs

def boldlike(exp, fname, fpath, rois=['V1'], featurematrices=None, valreps=2,
			valind=0, subj='S1', xfms='fullhead', wts=None, rho=0.8, std=1.0,
			ave=0.0, norm=True):
	"""
            Creates a response using a feature matrix from a particular stimulus. Puts activity
            in a particular ROI of a noisy brain. 
            Source provided by Mark L; also referenced his source code for generating fake data.
            github.com/marklescroart/fakedata
		:parameters:
			- exp: string. Experiment name. Used to name output files
				or to query bold responses in the scramble method.
			- fname : string. The name to save the NIFTI files as.
				Default is None.
			- fpath : string. The path where to store the resulting NIFTI files.
			- rois : list of strings. All the ROIs where the fake activity should be
				happening. Default is ['V1'].
			- featurematrices : list of feature matrices. The feature matrices to use
				when creating the fake responses (e.g feature matrix of each stimuli).
				Default is None.
			- valreps : integer. The number of validation responses to create.
				Default is 2 responses.
			- valind : integer. The index of the feature matrix in featurematrices
				that corresponds to the validation stimuli. Default is 0.
			- subj : string. The subject whose flatmap (in pycortex) to use.
				Default is 'S1'
			- xfms : string. The pycortex transform to use. Default is 'fullhead.'
			- wts : np.ndarray. The weights used to construct the fake response.
				Default is None.
			- rho : float. Correlation amount between the expected and actual response
				in the fake data. Default is 0.8.
			- std : float. The standard deviation of the gaussian noise in the brain outside
				the ROI of activity. Default is 1.0.
			- ave : flaot. The average of the gaussian noise in the brain outside the
				ROI of activity. Default is 0.0
			- norm : boolean. Indicates whether or not to normalize the activity created in
				the ROI. 
	"""
	newBRs = defaultdict(list)
	# Create the filename
	if fname is None:
		fname = exp + '_testR_boldlike_ROI{0}'.format(rois)
	valname = fname + '_val{0}.nii'
	fname = fname + '_iter{0}.nii'
	# Retrieve the mask for the ROI of activity
	mask = retrieveMasks(subj, xfms, rois)
	cnt, vcnt, nVox = 0, 0, len(np.nonzero(mask)[0])
	for stim in featurematrices:
		# Set iteration number for each stimulus matrix depending if valid. or train.
		if featurematrices.index(stim) == valind:
			reps, vflag = valreps, True
		else:
			reps, vflag = 1, False
		for j in range(reps):
			# Create neural activity for ROI
			BRnew, wts = createResponse(stim, nVox, wts=wts, rho=rho, norm=norm)
			# Put activity in the ROI of a noisy brain
			BRnew = applyMask(BRnew, mask, std=std, ave=ave)
			if vflag:
				BRnewI = savenewBR(BRnew, fpath, valname, cnt=vcnt)
				newBRs[valname.format(vcnt)].append(BRnewI)
				vcnt += 1
			else:
				BRnewI = savenewBR(BRnew, fpath, fname, cnt=cnt)
				newBRs[fname.format(cnt)].append(BRnewI)
				cnt += 1
	outfile = os.path.join(fpath, exp+'_testR_boldlike_ROI{0}.hdf5'.format(rois))
	save_table_file(outfile, dict(wts = wts))
	return newBRs, wts

def scrambleHelper(BR, chunklen):
	"""
	Scramble helper does the actually scrambling for scrambled fake data.
		:parameters:
			- BR : np.ndarray. Bold response to scramble.
			- chunklen : int. The chunk sizes to scramble
	"""
	allinds = range(BR.shape[0])
	# Find chunk length that divides the signal evenly
	while BR.shape[0] % chunklen != 0:
		chunklen -= 1
	# Group indices by chunklength
	indchunks = zip(*[iter(allinds)]*chunklen)
	# Shuffle indices
	shuffle(indchunks)
	indchunks = list(chain.from_iterable(indchunks))
	return BR[indchunks,:,:,:]

def savenewBR(BRnew, fpath, fname, cnt=None):
	"""
	Saves the newly created fake bold response.
		:parameters:
			- BRnew : np.ndarray. Fake bold data.
			- fpath : string. Directory to save fake data.
			- fname : string. Filename to save fake data under.
			- cnt : integer. If filenames are numbered.
				Default is None. 
	"""
	BRnewI = ni.Nifti1Image(BRnew, np.eye(4))
	if cnt:
		BRnewI.to_filename(os.path.join(fpath, fname.format(cnt)))
	else:
		BRnewI.to_filename(os.path.join(fpath, fname))
	return BRnewI

def addnoise(sig, std=None, use_sig=False):
	"""
	Adds noise to a signal.
		:parameters:
			- sig : np.ndarray. A signal to add noise to.
			- std : float. Standard deviation of the gaussian noise.
				Default is None.
			- use_sig : boolean. Whether or not to use the input signal's
				standard deviation value. Default is False.
	"""
	if std is None:
		if use_sig:
			std = np.std(sig, axis=0)
		else:
			std = random()
	noise = np.random.randn(*sig.shape)
	new = sig + noise * std
	return new

def loadShuffledIMS(exp, action):
	"""
	Queries and returns bold responses from docdb, after shuffling them.
		:parameters:
			- exp : string. Which experiment to query.
			- action : string. Which action to use in the query.
	"""
	dbi = docdb.getclient()
	ims = dbi.query(experiment_name=exp, generated_by_name=action)
	inds = range(len(ims))
	shuffle(inds)
	shuffIms = [ims[i] for i in inds]
	return shuffIms

def retrieveMasks(subj, xfms, rois):
	"""
	Retrieve the ROI mask that will be used.
		:parameters:
			- subj : string. Which subject's pycortex surface.
			- xfms : string. Which transforms for that subject.
			- rois : string list. Which ROI masks to retrieve. 
	"""
	masks = get_roi_mask(subj, xfms, roi=rois)
	allM = [masks[r] for r in rois]
	# If multiple ROIs are used, combines into one mask.
	return reduce(lambda x, y: x+y, allM)

def createResponse(stim, nVox, wts=None, rho=0.8, norm=True):
	"""
	Creates a response using a feature matrix from a particular stimulus.
		:parameters:
			- stim : np.ndarray. Feature matrix.
			- nVox : integer. The number of voxels the response needs
				to be.
			- wts : np.ndarray. The weights used to linearly combine
				the features (i.e. columns) of stim. Default is
				randomly generated weights.
			- rho : float. The correlation value between expected and
				actual signal. Default is 0.8.
			- norm : boolean. Indicates whether or not to normalize the neural
				activity. Default is True.
	"""
	t, nF = stim.shape[0], stim.shape[1]
	# If no weights or wrong sized weights, randomly generate weights
	if (wts is None) or (wts.shape != (nF, nVox)):
		print 'Either no weights provided or weights have incorrect\
		        dimensions. Will generate new weights...'
		wts =  np.random.ranf((nF, nVox))
	# Calculate expected response
	sig = np.dot(stim, wts)
	std = np.std(sig, axis=0)
	# Create the noise to add to the expected response
	noise = np.random.randn(t, nVox) * std
	R = rho * sig + np.sqrt(1-rho**2) * noise
	# Normalize the response
	if norm:
		R /= np.amax(R)
	return R, wts

def applyMask(sig, mask, std=1.0, ave=0.0, zdim=30, ydim=100, xdim=100):
	"""
	Create a full brain response.
		:parameters:
			- sig : np.ndarray. Neural response to go into a ROI.
				Correlated to a weighted version of a feature matrix.
			- mask : np.ndarray. Mask of the ROI where the signal will be going into.
			- std : float. The standard deviation of the gaussian noise in
				the baseline neural activity. Default is 1.0.
			- ave : float. The average of the gaussian noise in the baseline neural
				activity. Default is 0.0
			- zdim : interger. Z dimension size of the BOLD activity. Default is 30.
			- ydim : integer. Y dimension size of the BOLD activity. Default is 100.
			- xdim : integer. X dimension size of the BOLD activity. Default is 100.
	"""
	# generate a noisy brain
	newSig = np.random.randn(sig.shape[0], zdim, ydim, xdim) * std
	newSig += ave
	# insert actual signal in the ROI of the noisy brain
	for i in range(newSig.shape[0]):
		newSig[i, mask > 0] = sig[i, :]
	return newSig
