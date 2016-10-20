## adds noise to a pre-existing images and saves it to file

import nibabel as ni
import music.utils.generateTestResp as gtr
import os
import numpy as np

fpath='/auto/k7/lucine/projects/music/tmp'
fname = '20150608AN_testR_scramble_chunklen1_val{0}.nii'
valreps = 2


source = os.path.join(fpath,fname.format(0))
orig = ni.load(source).get_data()
print 'Loaded original validation data'
for i in range(1,valreps):
	print 'Adding noise...'
	newVal = gtr.addnoise(orig)
	print 'Added noise, now saving...'
	BRnewI = ni.Nifti1Image(newVal, np.eye(4))
	BRnewI.to_filename(os.path.join(fpath, fname.format(i)))
	print 'Saved.'
