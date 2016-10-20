import os
from collections import defaultdict
import docdb
import numpy as np
import tables
import logging
import hashlib
from util import save_table_file
import nibabel as ni
import cortex
# changed utils.util --> util for script to work

# changed story to song, not sure if it will work
logger = logging.getLogger("song.util.response_utils")


def load_response_imagedocs(experiments, fromaction="DetrendSGolay"):
    """Loads music responses from the given [experiments]. All images generated
    by the action [fromaction] will be used.

    This function creates a dictionary of (song name : list of responses).
    """
    # Get the interface to the database
    docdbi = docdb.getclient()

    # Create the output dictionary
    outdict = defaultdict(list)

    # For each experiment listed, fetch images for that experiment
    for exp in experiments:
        ims = sorted(docdbi.query(experiment_name=exp,
                                  generated_by_name=fromaction),
                     key=lambda im: im.block_number)

        # Add each image to the dictionary
        for im in ims:
            # The name of the sequence from the scanner
            imsong = im.SeriesDescription
            outdict[imsong].append(im)

    return outdict


# Following are FI's song data
def load_response_imagedocs_music_AN(sessions=1, usesg=True, unwarp=False):
    """Loads response images for AN."""
    if sessions == 1:
        exps = ["20150608AN"]
    elif sessions == 2:
        exps = ["20150618AN"]
    elif sessions == 3:
        exps = ["20150608AN", "20150618AN"]

    if usesg:
        fromaction = "DetrendSGolay"
    else:
        fromaction = "DetrendMedian"

    return load_response_imagedocs(exps, fromaction)

# Following are FI's song data
def load_response_imagedocs_speech_AN(sessions=1, usesg=True, unwarp=False):
    """Loads response images for AN."""
    if sessions == 1:
        exps = ["20150801AN"]
    elif sessions == 2:
        exps = ["20150801AN"]
    elif sessions == 3:
        exps = ["20150801AN"]

    if usesg:
        fromaction = "DetrendSGolay"
    else:
        fromaction = "DetrendMedian"

    return load_response_imagedocs(exps, fromaction)

# Following is for tonotopy pilot
def load_response_imagedocs_tonotopy_LO(sessions=1, usesg=True, unwarp=False):
    """Load response images for LO"""
    if sessions == 1:
        exps = ["20160330LO"]

    if usesg:
        fromaction = "DetrendSGolay"
    else:
        fromaction = "DetrendMedian"

    return load_response_imagedocs(exps, fromaction)

def load_response_imagedocs_tonotopy_AN(sessions=1, usesg=True, unwarp=False):
    """Load response images for AN"""
    if sessions == 1:
        exps = ["20160601AN"]

    if usesg:
        fromaction = "DetrendSGolay"
    else:
        fromaction = "DetrendMedian"

    return load_response_imagedocs(exps, fromaction)


# Following is for test response loading
def load_response_imagedocs_test(reading_dir, fnames, validation=None):
    """Loads the fake data for testing pipeline."""
    output = defaultdict(list)
    for fname in fnames:
        outfile = os.path.join(reading_dir, '{0}.nii'.format(fname))
        if 'val' in fname:
            output[validation].append(ni.load(outfile))
        else:
            output[fname].append(ni.load(outfile))
    return output

def dilate_mask(mask, ktype):
    """Dilates the given binary [mask] with the specified type of [kernel],
    which should be an integer [1, 2, 3] that is passed as the connectivity
    argument to scipy.ndimage.morphology.generate_binary_structure. See the
    docs of that function for details.
    """
    import scipy.ndimage.morphology as mo
    kernel = mo.generate_binary_structure(3, ktype)
    return mo.binary_dilation(mask, kernel)


def load_mcparams(respdict):
    """This function loads motion correction parameter estimates for the listed
    images."""
    mcparams = dict()
    for song, docs in respdict.items():
        logger.info("Loading mcparams for song {}..".format(song))
        mcdata = []
        for d in docs:
            if d.parent.generated_by_name == "MotionCorrectFSL":
                transforms = d.parent.generated_by.outputs["transforms"]
                params = transforms.get_params()
                mcdata.append(np.hstack(params))
            else:
                transforms = d.parent.generated_by.inputs["transform"][0]
                transforms = transforms.inputs["xfm1"][0].outputs["transforms"]
                params = transforms.get_params()
                mcdata.append(np.hstack(params))

        mcparams[song] = sum(mcdata) / len(mcdata)

    return mcparams


def load_responses(respdict, mask, cachedir="/auto/k8/loganesian/respcache/",
                   force_reload=True, multiseries="mean"):
    """This function caches the responses in an HDF5 file and reads them if the
    image IDs haven't changed.

    If there are multiple image series for a given stimulus, the images will
    either be averaged, if [multiseries] is "mean", or concatenated, if
    [multiseries] is "cat".
    """
    # Generate cache filename for this set of responses
    songnames = "".join(sorted(respdict.keys()))
    expnames = "".join(sorted([d[0].experiment_name
                               for d in respdict.values()]))
    nmask = str(mask.sum())
    resphash = hashlib.sha1(songnames+expnames+multiseries+nmask).hexdigest()
    cachefilename = os.path.join(cachedir, resphash+".hf5")

    # Check if the cache file exists:
    if os.path.exists(cachefilename) and not force_reload:
        logger.info("Loading responses from cache file..")
        # Try to load the data from it
        try:
            rtf = tables.openFile(cachefilename)
            resparrs = dict()
            for song in respdict.keys():
                resparrs[song] = rtf.getNode("/"+song).read()
            rtf.close()
            return resparrs
        except Exception, e:
            logger.error("Failed to load data from cache ({}), loading\
                          normally..".format(str(e)))

    # Now either the cache file doesn't exist or there was an error
    # Either way load the data normally, then save it
    logger.info("Loading responses normally..")
    resparrs = _load_responses(respdict, mask, multiseries)

    logger.info("Caching responses..")
    save_table_file(cachefilename, resparrs)

    return resparrs

def load_responses_test(respdict, mask, cachedir="/auto/k8/loganesian/respcache/",
                   force_reload=True, multiseries="mean"):
    """This is a version of load_responses() from above, modified to work with
    fake data. Only difference is that experimentnames is omitted.
    """
    # # Generate cache filename for this set of responses
    songnames = "".join(sorted(respdict.keys()))
    nmask = str(mask.sum())
    resphash = hashlib.sha1(songnames+multiseries+nmask).hexdigest()
    cachefilename = os.path.join(cachedir, resphash+".hf5")

    # # Check if the cache file exists:
    # if os.path.exists(cachefilename) and not force_reload:
    #     logger.info("Loading responses from cache file..")
    #     # Try to load the data from it
    #     try:
    #         rtf = tables.openFile(cachefilename)
    #         resparrs = dict()
    #         for song in respdict.keys():
    #             resparrs[song] = rtf.getNode("/"+song).read()
    #         rtf.close()
    #         return resparrs
    #     except Exception, e:
    #         logger.error("Failed to load data from cache ({}), loading\
    #                       normally..".format(str(e)))

    # Now either the cache file doesn't exist or there was an error
    # Either way load the data normally, then save it
    logger.info("Loading responses normally..")
    resparrs = _load_responses(respdict, mask, multiseries)

    logger.info("Caching responses..")
    save_table_file(cachefilename, resparrs)

    return resparrs


def _load_responses(respdict, mask, multiseries):
    """Loads actual response data from the dictionary of lists of response image
    documents, [respdict].
    """
    # Generate cache filename for this set of responses

    resps = dict()
    for song, docs in respdict.items():
        logger.info("Loading response data for song {}..".format(song))
        datas = [d.get_data()[:, mask] for d in docs]
        if multiseries == "mean":
            resps[song] = sum(datas)/len(datas)
        elif multiseries == "cat":
            resps[song] = np.vstack(datas)

    return resps

def selectROI(respdict, ROIS, surface, experiment, mask, multiseries="mean"):
    """Select a particular ROI to do the regression on"""

    # selected = dict()
    # for song, data in respdict.items():
    #     newData = np.zeros(data.shape)
    #     newData[:,ROImask] = data[:,ROImask]
    #     selected[song] = newData

    ROImask = cortex.get_roi_masks(surface, experiment, ROIS)[0]
    # ROImask = ROImask[0] # uncessary line
    # ROImask = ROImask != 0 # original
    ROImask = ROImask == 0 # trying something out
    resps = dict()
    for song, docs in respdict.items():
        logger.info("Loading response data for song {}..".format(song))
        datas = []
        for d in docs:
            tmp = d.get_data()
            tmp[:, ROImask] = 0
            datas.append(tmp[:, mask])
        if multiseries == "mean":
            resps[song] = sum(datas)/len(datas)
        elif multiseries == "cat":
            resps[song] = np.vstack(datas)

    return resps

