import docdb
# import os
import sys

docdbi = docdb.getclient()
if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
    print "Experiment_name from command line"
else:
    experiment_name_one = "20150608AN"
    experiment_name_two = "20150618AN"

# action_type = "TemporalMean" # need to use coregister to check if aligned
action_type = "DetrendSGolay" # can see more alignment with Detrended images

tmean_one = docdbi.query(experiment_name=experiment_name_one,
                     generated_by_name=action_type, block_number=0)[0]
tmean_two = docdbi.query(experiment_name=experiment_name_two,
                     generated_by_name=action_type, block_number=0)[0]
# tmean.fslview()
# nifti_dir = "/auto/k8/fatma/projects/music/fmri/20140428FI/"
# img_name = "AN_mean_pairwisecorr.nii"
# tmean.fslview(with_ims=[os.path.join(nifti_dir, img_name)])

tmean_one.fslview(with_ims=[tmean_two])
