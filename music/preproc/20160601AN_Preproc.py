#!/usr/bin/python

import docdb
from pipelines.generic import generic_preproc

session = 1
docdbi = docdb.getclient()
experiment_name = "20160601AN"
runtemplate = "/auto/data/archive/mri/"+experiment_name+"/{0}_{1}"
if session == 1:
    #runs = [8, 11, 12, 15, 16, 19, 20, 23, 24]  # 8 and 24 are validation
    runs = [24] # messed up; ran script w/o adding one more soundfile in soundfiles
#    soundfiles = ["ep2d_ton_TE30ms",
#                  "ep2d_ton_TE30ms",
#                  "ep2d_ton_TE30ms",
#                  "ep2d_ton_TE30ms",
# 		   "ep2d_ton_TE30ms",
# 		   "ep2d_ton_TE30ms",
#		   "ep2d_ton_TE30ms",
#		   "ep2d_ton_TE30ms"]
    soundfiles = ['ep2d_ton_TE30ms']
elif session == 2:
    pass

#ref_image = None

# because need to do 9th run individually, need ref_image
ref_image = docdbi.query(experiment_name="20160601AN", generated_by_name="TemporalMean", block_number=0)[0]
ref_image = ref_image.inputform


dicomdirs = [runtemplate.format(song, run) for song, run in zip(soundfiles,
                                                                runs)]
actgraph = generic_preproc(docdbi,
                           dicomdirs,
                           experiment_name,
                           reference=ref_image,
                           do_brainmask=False,
                           slicetime=False,
                           smooth=None,
                           detrend="sg",
                           do_tsdiag=True)

actgraph.run(docdbi, local=True, simple_local=True)
