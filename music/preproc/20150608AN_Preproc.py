#!/usr/bin/python

import docdb
from pipelines.generic import generic_preproc

session = 1
docdbi = docdb.getclient()
experiment_name = "20150608AN"
runtemplate = "/auto/data/archive/mri/"+experiment_name+"/{0}_{1}"
if session == 1:
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
    experiment_name = "20150618AN"
    runs = [5, 9, 13, 17, 21, 25, 29, 33] # 9 and 29 are validation (beethoven) 
    soundfiles = ["Bach_BWV875_0102_Beethoven_Op27No1_03",
                  "Beethoven_WoO080", 
                  "Brahms_Op005_01", 
                  "Chopin_Op010_03_Op028_04_Op029", 
                  "Chopin_Op010_04_Op048No1", 
                  "Haydn_HobXVINo52_01", 
                  "Beethoven_WoO080",
                  "Ravel_JeuxDEau_Skyrabin_Op008No8"]

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

actgraph.run(docdbi, local=True, simple_local=True)
