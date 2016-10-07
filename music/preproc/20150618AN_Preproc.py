#!/usr/bin/python

import docdb
from pipelines.generic import generic_preproc

session = 2
docdbi = docdb.getclient()
if session == 1:
    experiment_name = "20150608AN"
    runs = [5, 9, 13, 17, 19, 23, 27, 29, 33, 37]  # 9 and 33 are validation
    soundfiles = ["Beethoven_Op027_No101",
                  "Beethoven_WoO080",
                  "Beethoven_Op031No201",
                  "Brahms_Op010No1",
                  "Brahms_Op010No2",
                  "Chopin_Op026No1",
                  "Chopin_Op026No2",
                  "Chopin_Op066",
                  "Beethoven_WoO080",
                  "Rachmaninoff_Op03603_Op03901"]
elif session == 2:
    #PLEASE UPDATE HERE
    experiment_name = "20150618AN"
    runs = [5, 9, 13, 17, 21, 25, 29, 33] # 9 and 29 are validation (beethoven)
    soundfiles = ["Bach_BWV875_0102_Beethoven_Op27No1_03",
                  "Beethoven_WoO080",
                  "Brahms_Op005_01",
                  "Chopin_Op010_03_Op028_04_Op29",
                  "Chopin_Op10_04_Op48No1",
                  "Haydn_HobXVINo52_01",
                  "Beethoven_WoO080",
                  "Ravel_JeuxDeau_Skyrabin_Op008No8"]

runtemplate = "/auto/data/archive/mri/"+experiment_name+"/{0}_{1}"
dicomdirs = [runtemplate.format(song, run) for song, run in zip(soundfiles,
                                                                runs)]

if session == 1:
	# The temporal mean of the first run will be automatically used as the reference
	ref_image = None
elif session == 2:
	# Use the temporal mean of the first run from the first session as the reference
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
