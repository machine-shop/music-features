import cortex as cx
import docdb
import numpy as np
import nibabel as nib

experiment_name = "20150618AN"
surface = "ANfs"
fname = '/auto/k7/lucine/projects/music/tmp/20150618_tmean.nii'
docdbi = docdb.getclient()

# What's the point of ims, where even is it used?
ims = sorted(docdbi.query(experiment_name=experiment_name,
                          generated_by_name="DetrendSGolay"),
             key=lambda im: im.block_number)

# Align subjects temporal mean of the first run (block_number=0) to the
# pycortex surface-space

# ORIGINAL
# tmean = docdbi.query(experiment_name=experiment_name,
#                     generated_by_name="TemporalMean", block_number=0)[0]

# TEST
dtrend = docdbi.query(experiment_name=experiment_name,
                      generated_by_name="DetrendSGolay", block_number=0)[0]
tmp = dtrend.get_data()
tmean = tmp.mean(0)
tmean = nib.Nifti1Image(np.transpose(tmean), np.eye(4))
tmean.to_filename(fname)

# Subjects' transformations are stored in "/auto/k2/share/pycortex_store/"

# ORIGINAL
# cx.align.automatic(surface, experiment_name, tmean.path)

# TEST
cx.align.automatic(surface, experiment_name, fname)

# Check alignment and eventually make adjustments (save only using the buttons
# until bug is fixed!
# cx.align.manual(surface, experiment_name)
# Use decimate=True option for visualization purpose. Uses less dots to
# estimate along the surface.
# cx.align.manual(surface, experiment_name, decimate=True)

# Check which surfaces exist
print cx.db.ANfs.transforms
xfm = cx.db.get_xfm(surface, experiment_name)
xfm_val = xfm.xfm  # Gives the 4D transformation matrix
