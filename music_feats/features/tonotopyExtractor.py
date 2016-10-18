import numpy as np
import os

__all__ = ['tonotopy_labels',
		   'averageDownsample']

def tonotopy_labels(fname, fpath='/auto/k8/loganesian/stimuli/tonotopy/uncalibrated_wavs', numF=10):
	"""
	Use the labels generated when creating the tonotopy stimuli to create a binary feature matrix.
		:parameters:
            - fname : the name of the stimulus of which you wish to load the labels.
			- fpath : the path where the tonotopy files are saved.
				Default : '/auto/k8/loganesian/stimuli/tonotopy/uncalibrated_wavs'
			- numF : the number of tonal centers used in the tonotopy experiment.
		:returns:
			- data_matrix : np.ndarray. [shape=(n, numF)] Binary matrix denoting which
					 	    frequency center correponds to which sample.
			- labels : np.ndarray. [shape=(n,)] The actual labels for the sound file.
			- freqs : np.ndarray. [shape=(numF,)] The center frequencies used for the localizer.
					  Computed: np.logspace(np.log10(150), np.log10(9600), num=numF)
	"""
	full_fname = os.path.join(fpath, fname)
	labels = np.load('{0}_Labels.npy'.format(full_fname)) # load the labels
	# generate the frequencies
	freqs = list(np.logspace(np.log10(150), np.log10(9600), num=numF))
	data_matrix = np.zeros((numF, len(labels))) # initialize the sub-matrix
	for i in range(numF):
	# binary categories; mark where f in freqs happens
		data_matrix[i, labels-(labels%1) == freqs[i]-(freqs[i]%1)] = 1
	return data_matrix, labels, freqs

def averageDownsample(vals, newlen=None, overlapsize=1):
        """
        Downsample feature matrix by taking a moving window overage of some overlap. Primarily used
		with the tonotopy labels feature.
            :parameters:
                - vals : np.ndarray. The values to be downsampled. (size: time x feature num)
                - newlen : integer. The new length the downsampled signal should be.
                           Default is None, reutrning original values untouched.
                - overlapsize: integer. The amount of overlap when computing the averages.
                           Default is 1.
        """
        if newlen is None:
            print('New length value not provided, will not average and downsample...')
            return vals
        newarr = np.zeros((newlen, vals.shape[1]))
        winsize = vals.shape[0]//newlen + overlapsize
        padWidth = ((0, winsize - vals.shape[0] % newlen), (0,0))
        # make sure there is enough array to actually take window averages over
        toDownsample = np.pad(vals, pad_width=padWidth, mode='reflect')
        for i in range(newlen):
            start = i*(winsize-overlapsize)
            end = start + winsize
            newarr[i,:] = np.mean(toDownsample[start:end,:], axis=0)
        return newarr
