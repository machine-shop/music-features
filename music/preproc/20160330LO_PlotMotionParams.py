import os
import docdb
docdbi = docdb.getclient()

exps = ["20160330LO"]
outdir = "/auto/k8/loganesian/projects/music/figures"
if not os.path.exists(outdir):
    os.mkdir(outdir)

for exp in exps:
    print exp
    expdir = os.path.join(outdir, exp, "motionparams")
    if not os.path.exists(expdir):
        os.makedirs(os.path.join(expdir))

    mcims = sorted(docdbi.query(experiment_name=exp,
                                generated_by_name="MotionCorrectFSL"),
                   key=lambda im: im.block_number)
    for mcim in mcims:
        runname = mcim.SeriesDescription
        runnr = mcim.SeriesNumber
        if runnr == 9.:
            runname = runname+'_1'
        if runnr == 33.:
            runname = runname+'_2'
        print runname
        filename = os.path.join(expdir, runname+".png")
        mcim.generated_by.outputs["transforms"].plot_params(filename)
