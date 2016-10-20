import os
import numpy as np
from collections import defaultdict


class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.0045):
        """Loads data from [trfilename], should be output from stimulus
        presentation code.
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr

        if trfilename is not None:
            self.load_from_file(trfilename)

    def load_from_file(self, trfilename):
        """Loads TR data from report with given [trfilename].
        """
        # Read the report file and populate the datastructure
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label == "sound-start":
                self.soundstarttime = time

            elif label == "sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))

        # Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes > (itrtimes.mean()*1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            # Insert new TR where it was missing..
            newtrtime = self.trtimes[btr]+self.expectedtr
            newtrs.append((newtrtime, btr))

        for ntr, btr in newtrs:
            self.trtimes.insert(btr+1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR.
        """
        self.trtimes = list(np.arange(ntrs)*self.expectedtr)

    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound.
        """
        return np.array(self.trtimes)-self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run.
        """
        return np.diff(self.trtimes).mean()

# TODO need to update the root default to something relevant
def load_trfiles(respdict, root="/auto/data/archive/mri/stimreports/"):
    """Loads a dictionary of TRFiles for the given responses.
    """
    trdict = dict()

    for song, resps in respdict.items():
        trfs = []
        # Sort the responses by experiment
        expresps = defaultdict(list)
        for r in resps:
            expresps[r.experiment_name].append(r)

        for expname, er in expresps.items():
            # Fix expname if it has a suffix
            if "-" in expname:
                expname = expname.split("-")[0]

            if len(er) > 1:
                # Need to load multiple TRFiles
                for r in er:
                    bnum = r.block_number
                    fname = "{0}-{1}-{2}.report".format(expname, song, bnum)
                    trf = TRFile(os.path.join(root, fname))
                    trfs.append(trf)
            else:
                # Just load the one TRFile
                fname = "{0}-{1}.report".format(expname, song)
                trf = TRFile(os.path.join(root, fname))
                trfs.append(trf)

        trdict[story] = trfs

    return trdict


def load_generic_trfiles(respdict,
                         root="/auto/k1/huth/text/story/stimreports/generic"):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the
    session in which the data was collected.. this should be fine) for the
    given responses."""
    trdict = dict()

    for stimulus, resps in respdict.items():
        try:
            trf = TRFile(os.path.join(root, "{}.report".format(stimulus)))
            trdict[stimulus] = [trf]
        except Exception, e:
            print e

    return trdict


def load_generic_trfiles_fi(stimuli, subject, root="data/trfiles"):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the
    session in which the data was collected.. this should be fine) for the
    given stimuli."""
    trdict = dict()

    for stimulus in stimuli:
        try:
            fname = "{0}_{1}.report".format(stimulus, subject)
            trf = TRFile(os.path.join(root, fname))
            trdict[stimulus] = [trf]
        except Exception, e:
            print e

    return trdict
