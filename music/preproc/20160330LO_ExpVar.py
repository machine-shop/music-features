#!/usr/bin/python
import docdb
import numpy as np
import os
from docdb.actions import CreateGroup, ExpVariance, FStats
from fmriutils.util import image_statistic, voxel_correlation

docdbi = docdb.getclient()
experiment_name = "20160330LO"
cropval = [5, 334]
result_dir = "/auto/k8/loganesian/projects/music/results/{}".format(experiment_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# Create an empth ActionGraph
actgraph = docdb.actions.ActionGraph()

# Get the Validation image docs
ims = sorted(docdbi.query(experiment_name=experiment_name,
                          generated_by_name="DetrendSGolay",
                          SeriesDescription="Beethoven_WoO080"), # erm....TODO
             key=lambda im: im.block_number)

# Get the action that generated this ImageDoc
ims_action = [im.inputform for im in ims]

# Create a group of actions. The output of these actions will further be used
# in the statistics.
ims_group = CreateGroup(inputs=ims_action, params={}, dbinterface=docdbi)

# Fstatistics action
print "Fstatistics \n"
fstat_action = FStats(inputs=dict(data=(ims_group, "group")),
                      params=dict(crop=cropval, blocklength=None,
                      blockorder=None), dbinterface=docdbi)

# Add to the ActionGraph
actgraph.add([ims_group, fstat_action])

# Explainable variance action
print "Explainable variance\n"
ev_action = ExpVariance(inputs=dict(data=(ims_group, "group")),
                        params=dict(crop=cropval, blocklength=None,
                        blockorder=None), dbinterface=docdbi)

# Add to the ActionGraph
actgraph.add([ims_group, ev_action])

# Run the ActionGraph
actgraph.run(docdbi, local=True, simple_local=True)

# Run pairwise correlation
corr = images_statistic(ims, voxel_correlation, limits=cropval)
filename = "{0}/{1}_VoxelCorr.npz".format(result_dir, experiment_name)
np.savez(filename, corr=corr)

# ev = docdbi.query(experiment_name=experiment_name,
#                   generated_by_name="ExpVariance")
# ev = ev[0].get_data()
# fstat = docdbi.query(experiment_name=experiment_name,
