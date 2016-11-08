import numpy as np
import cortex
#from regression_code.huth.npp import mcorr
from music_feats.utils.viz import separate_model_weights

__all__ = ['STA',
            'spectralCentroid',
            'createVolume',
            'averageDelays',
                ]


def STA(delRstim, zRresp, delPstims, zPresps, ndelays, modeldims):
    """Function estimates model weights using a spike-triggered average
    model. Weights are cross-validated and the results returned.

    Parameters
    ----------
    delRstim : np.ndarray
        training - stimulus feature time series
    zRresp : np.ndarray
        training - voxel responses
    delPstims : list of np.ndarry
        validation - stimulus feature time series
    zPresps : list of np.ndarray
        validation - voxel responses
    ndelays : python list of ints
        number of delays to use in the model
    modeldims : python list of ints, 
        dimensions of the submodels as a list

    Returns
    -------
    np.ndarray, np.ndarray, list, list
        STA_W : estimated weights, valcorr_STA : validation correlation values (full model)
        submodelwts_STA : wts for each submodel, valmodelcorr_STA : list of prediction correlcation
        values for each submodel

    """
    STA_W = np.dot(delRstim.T, zRresp)
    STA_valpreds = [np.dot(delPstim, STA_W) for delPstim in delPstims]
    
    valcorr_STA = np.vstack([mcorr(np.dot(delPstim, STA_W), zPresp)
                             for (delPstim, zPresp) in zip(delPstims, zPresps)])
    
    submodelwts_STA = separate_model_weights(STA_W, ndelays, modeldims)
    valmodelcorr_STA = []

    for delPstim, zPresp in zip(delPstims, zPresps):
        sepdelPstim = separate_model_weights(delPstim.T, ndelays, modeldims)
        valmodelcorr_STA.append(np.vstack([mcorr(np.dot(s.T, w), zPresp)
                                          for (s,w) in
                                          zip(sepdelPstim, submodelwts_STA)]))

    return STA_W, valcorr_STA, submodelwts_STA, np.nan_to_num(valmodelcorr_STA)

def spectralCentroid(W):
    """Function used for calculating spectral centroid of the weights. Computes the
    centroid of the weights (rows) for each voxel (column). Frequency approximated
    by index of the weights.

    Parameters
    ----------
    W : np.ndarray
        [(num_channels, num_voxels)].

    Returns
    -------
    np,ndarray,
        SC :[(num_voxels)] One value per voxel correspond to the
        centroid of frequency indices.
 
    """
    rectified_w = W.copy()
    rectified_w[rectified_w < 0] = 0
    freq_inds = np.arange(rectified_w.shape[0])[:,None] # indices of weights used
    total_weights = np.sum(rectified_w, axis=0)
    return np.nan_to_num(np.sum(freq_inds * rectified_w, axis=0)/total_weights)

def createVolume(data, mask, surface, xfms, vmin=None, vmax=None, factor=0.95):
    """Helper function for creating a cortex volume for visualization.

    Parameters
    ----------
    data : np.ndarray
        The data to visualize.
    mask : np.ndarray
        Mask to use to transform.
    surface : TYPE 
        surface to use.
    xfms :  TYPE
        transform to use.
    vmin : float
        Default: Min of data.
    vmax : float
        Default: Max of data.
    factor : float
        Factor of vmax to use for visualization purposes.
             Default 0.95.

    Returns
    -------
    cortex volume
        
    """
    volData = np.zeros(mask.shape)
    volData[mask>0] = data
    
    if vmin is None:
        vmin = np.min(volData)
    if vmax is None:
        vmax = np.max(volData) * factor
    else:
        vmax = factor * vmax
    
    return cortex.Volume(volData, surface, xfms, vmin=vmin, vmax=vmax)

def averageDelays(wt, modeldims, delays=4):
    """Compute the average weight across delays. Assumes there are wt.shape[0]/delays
    features overall (sub-models of dimension 1). Used for tonotopy analysis.

    Parameters
    ----------
    wt : np.ndarray 
        [(num_channels*delays, num_voxels)]. Delayed model weights.
    modeldims: list
        List of submodel dimensions (should be list of all 1s).
    delays : int
        Number of delays used. Default: 4.

    Returns
    -------
    int
        Average of delays

    """
    sep_fullwts = separate_model_weights(wt, delays, modeldims)
    
    avesubmdlwts = []

    for k in range(len(sep_fullwts)):
        avg = sep_fullwts[k].mean(0)
        avesubmdlwts.append(avg)

    avesubmdlwts_stk = np.vstack(avesubmdlwts)
    return avesubmdlwts_stk
