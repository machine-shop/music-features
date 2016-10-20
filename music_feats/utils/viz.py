import numpy as np
import operator

def separate_voxel_model_weights(vwt, ndelays, modeldims):
    """Takes a voxel weight vector [vwt] and separates out the
    weights for each model. Returns a list of DxNi matrices, where
    D is [ndelays] and Ni is the number of dimensions in model i,
    given in [modeldims].
    """
    rwt = vwt.reshape((ndelays, -1)) ## Separate delays
    srwt = np.split(rwt, np.cumsum(modeldims)[:-1], 1) ## Split models
    return srwt

# used in the model script; separate weights
def separate_model_weights(vwts, ndelays, modeldims):
    """Takes an N x M matrix of voxel weights [vwts] (where N is the
    total number of features and M is the number of voxels) and
    separates out the weights for each separate model.

    Returns a list of (D*Ni) x M matrices, where D is the number of
    delays and Ni is the number of features in model i.
    """
    ## Modelnums is an array of length N, where the i'th entry contains
    ## the index of the model that has the i'th feature.
    modelnums = np.tile(np.repeat(range(len(modeldims)), modeldims), ndelays)
    return [vwts[modelnums==mi,:] for mi in range(len(modeldims))]

def undelay_model_weights(vwts, ndelays):
    return reduce(operator.add, np.split(vwts/ndelays, ndelays))

## DIFFERENT PROJECT ##
phonemes = ['AA', 'AE','AH','AO','AW','AY','B','CH','D',
    'DH', 'EH', 'ER',   'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH',
    'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
wordclasses = ["noun",
               "conj/prep",
               "adjective",
               "prep",
               "verb",
               "ppron+nom",
               "adverb",
               "qualifier",
               "conj",
               "adjective",
               "ppron+acc",
               "sub conj",
               "adv/prep",
               "said",
               "conj",
               "prequant",
               "aux verb",
               "determiner",
               "aux/adverb",
               "past verb",
               "sstart",
               "send"][:20]
