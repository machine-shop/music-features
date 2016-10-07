#!/usr/bin/python

import docdb
from pipelines.generic import generic_preproc

session = 1
docdbi = docdb.getclient()
if session == 1:
    experiment_name = "20140729ML"
    runs = [9, 13, 17, 21, 25, 29, 
              33, 37, 41]  
              # 9, 17, 25, 33, 41 are validation
    soundfiles = ["holmesval",
                  "holmes01",
                  "holmesval",
                  "holmes02",
                  "holmesval",
                  "holmes03",
                  "holmesval",
                  "holmes04",
                  "holmesval"]
elif session == 2:
    pass

runtemplate = "/auto/data/archive/mri/"+experiment_name+"/{0}_{1}"
dicomdirs = [runtemplate.format(song, run) for song, run in zip(soundfiles,
                                                                runs)]

actgraph = generic_preproc(docdbi,
                           dicomdirs,
                           experiment_name,
                           reference=None,
                           do_brainmask=True,
                           slicetime=False,
                           smooth=None,
                           detrend="sg",
                           do_tsdiag=True)

actgraph.run(docdbi, local=True, simple_local=True)  # include simple_local for debugging
