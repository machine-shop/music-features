#!/usr/bin/python

import docdb
from pipelines.generic import generic_preproc

session = 1
docdbi = docdb.getclient()
if session == 1:
    # experiment_name = "20150814AN"
    experiment_name = "20150814AN_01"
    runs = [5, 9, 13, 17, 21, 25, 29,
           33, 35, 51, 53] 
    soundfiles = ["howtodraw",
                  "3x_val3min_3xsilence",
                  "life",
                  "myfirstdaywiththeyankees",
                  "naked",
                  "wheretheressmoke",
                  "undertheinfluence",
                  "AN_Ashtray",
                  "AN_Ashtray",
                  "AN_Ashtray",
                  "AN_Ashtray"]
elif session == 2:
    pass

# runtemplate = "/auto/data/archive/mri/"+experiment_name+"/{0}_{1}"
runtemplate = "/auto/data/archive/mri/"+"20150814AN"+"/{0}_{1}"
dicomdirs = [runtemplate.format(song, run) for song, run in zip(soundfiles,
                                                                runs)]

if session == 1:
	# The temporal mean of the first music session used as reference
    ref_image = docdbi.query(experiment_name="20150608AN", generated_by_name="TemporalMean",
                 block_number=0)[0]
    ref_image = ref_image.inputform
elif session == 2:
	# The temporal mean of the first music session used as reference
    ref_image = docdbi.query(experiment_name="20150608AN", generated_by_name="TemporalMean",
 		          	 block_number=0)[0]
    ref_image = ref_image.inputform

actgraph = generic_preproc(docdbi,
                           dicomdirs,
                           experiment_name,
                           reference=ref_image,
                           do_brainmask=True,
                           slicetime=False,
                           smooth=None,
                           detrend="sg",
                           do_tsdiag=True)

actgraph.run(docdbi, local=True, simple_local=True)  # include simple_local for debugging
