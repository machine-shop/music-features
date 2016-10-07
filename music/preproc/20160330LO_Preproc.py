#!/usr/bin/python

import docdb
from pipelines.generic import generic_preproc

session = 1
docdbi = docdb.getclient()
experiment_name = "20160330LO"
runtemplate = "/auto/data/archive/mri/"+experiment_name+"/{0}_{1}"
if session == 1:
    runs = [12, 13, 16, 17, 20, 21, 24, 25]  # no validation?
    soundfiles = ["ep2d_ton_TE30ms",
                  "ep2d_ton_TE30ms",
                  "ep2d_ton_TE30ms",
                  "ep2d_ton_TE30ms",
		  "ep2d_ton_TE30ms",
		  "ep2d_ton_TE30ms",
		  "ep2d_ton_TE30ms",
		  "ep2d_ton_TE30ms"]
elif session == 2:
    pass

dicomdirs = [runtemplate.format(song, run) for song, run in zip(soundfiles,
                                                                runs)]

actgraph = generic_preproc(docdbi,
                           dicomdirs,
                           experiment_name,
                           reference=None,
                           do_brainmask=False,
                           slicetime=False,
                           smooth=None,
                           detrend="sg",
                           do_tsdiag=True)

actgraph.run(docdbi, local=True, simple_local=True)
