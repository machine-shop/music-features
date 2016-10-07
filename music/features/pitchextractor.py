import numpy as np
import pretty_midi
from collections import defaultdict

__all__ = ['BinnedNote',
          'calculatebintime',
		  'pitchSelectionHeuristic',
		  'pitchExtraction']

null_pitch = -1 # when there is no pitch present, assign -1

class BinnedNote(object):
	"""
	BinedNote object used to keep track of the pitch, velocity, and duration of
	the note that represents a particular time bin.
	"""
	def __init__(self, pitch, velocity, duration):
		self.pitch = pitch
		self.velocity = velocity
		self.duration = duration

def calculatebintime(starttime, endtime, binsize):
	"""
	Computes the 'start time' of the bin in which a particular note needs to be placed in.
	'Start time' is computed using the notes start and end time and the binsize to be used.
		:paramters:
			- starttime : float. The start time of the note.
			- endtime : float. The end time of the note.
			- binsize : float. The size of the bins used (in seconds).
	"""
	startbin = starttime//binsize * binsize
	if endtime >= startbin + binsize:
		return startbin, True
	else:
		return startbin, False

def pitchSelectionHeuristic(timebin, factor=0.05):
	"""
	A heuristic for selecting the 'representative' pitch of a time bin.
	Current heuristic: select the pitch that is the highest in the time bin OR
	pick the second highest pitch, if its duration is greather than factor * duration of the
	highest pitch.
		:parameters:
			- timebin : list. List of notes in a particular time bin.
			- factor : float. The factor by which to multiply the duration of the highest pitch in
				time bin, when trying to decide between the highest and second highest pitch.
		:returns:
			- pitch : float. The pitch that is the representative pitch of that time bin.
	"""
	if len(timebin) == 0:
		# if empty return nothing
		return null_pitch
	elif len(timebin) == 1:
		return timebin[0].pitch
	else:
		pitches = []
		velocities = []
		durations = []
		for note in timebin:
			pitches.append(note.pitch)
			velocities.append(note.velocity)
			durations.append(note.duration)
		maxpitch = max(pitches)
		maxind = pitches.index(maxpitch)
		pitches.remove(maxpitch)
		secondmaxpitch = max(pitches)
		pitches.insert(maxind, maxpitch)
		secondmaxind = pitches.index(secondmaxpitch)
		if durations[maxind] > durations[secondmaxind]:
			return maxpitch
		elif durations[maxind] == durations[secondmaxind]:
			if velocities[maxind] >= velocities[secondmaxind]:
				return maxpitch
			else:
				return secondmaxpitch
		else:
			difference = durations[secondmaxind]-durations[maxind] 
			if durations[maxind]*factor >= difference:
				return maxpitch
			else:
				return secondmaxpitch

def pitchExtraction(midi_file, num_instruments=1, binsize=2.0, **kwargs):
	"""
	Extracts the pitches from the midi file. Groups pitches within every binsize
	seconds according to the note start and end times. Then a 'representative' pitch
	for that particular bin is selected according to a pitch heursitc as defined in
	pitchSelectionHeuristic().
		:usage:
			>>> # Extract pitches of some midifile
			>>> pitches = pitchExtraction('sample.mid', num_instruments=1, binsize=2.0, factor=0.05)
			
		:parameters:
			- midi_file : string. The midi file from which to extract the pitches.
			- num_instrument : integer. The number of instruments in this midi file.
				Default: 1 instrument. WARNING: currently, as implemented, function ONLY
				works for 1 instrument.
			- binsize : float. The size of the time bins to be used when binning the notes
				together.
			- **kwargs : arguments to be used by pitchSelectionHeuristic().

		:returns:
			- extractedPitches : np.ndarray [shape=(1,n)] of 'representative' pitches for every
				binsize time units.
	"""
	if num_instruments > 1:
		print "Function hasn't been designed to work with more\
		       than one instrument; quiting now"
		return 
	
	midi_data = pretty_midi.PrettyMIDI(midi_file)
	instrument = midi_data.instruments[0]

	binnedNotes = defaultdict(list)
	extractedPitches = []
	
	for time in np.arange(0, midi_data.get_end_time(), binsize):
		binnedNotes[time] = []

	for note in instrument.notes:
		thisbin, nextbin = calculatebintime(note.start, note.end, binsize)
		if nextbin:
			consecbin = thisbin + binsize
			duration = consecbin - note.start
			binnedNotes[consecbin].append(BinnedNote(note.pitch,
				                                     note.velocity,
				                                     note.end - consecbin))
		else:
			duration = note.end - note.start
		binnedNotes[thisbin].append(BinnedNote(note.pitch, note.velocity, duration))

	for time in binnedNotes.keys():
		pitch = pitchSelectionHeuristic(binnedNotes[time], **kwargs)
		extractedPitches.append(pitch)

	return np.asarray(extractedPitches)
