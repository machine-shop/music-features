import cortex as cx
import docdb

experiment_name = "20140729ML"
surface = "MLfs2"
docdbi = docdb.getclient()
ims = sorted(docdbi.query(experiment_name=experiment_name,
                          generated_by_name="DetrendSGolay"),
             key=lambda im: im.block_number)

# Align subjects temporal mean of the first run (block_number=0) to the
# pycortex surface-space

tmean = docdbi.query(experiment_name=experiment_name,
                     generated_by_name="TemporalMean", block_number=0)[0]

# Subjects' transformations are stored in "/auto/k2/share/pycortex_store/"

cx.align.automatic(surface, experiment_name, tmean.path)

# Check alignment and eventually make adjustments (save only using the buttons
# until bug is fixed!
# cx.align.manual(surface, experiment_name)
# Use decimate=True option for visualization purpose. Uses less dots to
# estimate along the surface.
# cx.align.manual(surface, experiment_name, decimate=True)

# Check which surfaces exist
print cx.db.MLfs2.transforms
xfm = cx.db.get_xfm(surface, experiment_name)
xfm_val = xfm.xfm  # Gives the 4D transformation matrix
