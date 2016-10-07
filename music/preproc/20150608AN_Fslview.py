import docdb
# import os
import sys

docdbi = docdb.getclient()
if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
    print "Experiment_name from command line"
else:
    experiment_name = "20150608AN"

# action_type = "TemporalMean"
action_type = "TemporalMean"
tmean = docdbi.query(experiment_name=experiment_name,
                     generated_by_name=action_type, block_number=0)[0]
tmean.fslview()
# nifti_dir = "/auto/k8/fatma/projects/music/fmri/20140428FI/"
# img_name = "AN_mean_pairwisecorr.nii"
# tmean.fslview(with_ims=[os.path.join(nifti_dir, img_name)])
