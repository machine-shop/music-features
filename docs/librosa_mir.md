|  Feature                |  Librosa      |   MIR Toolbox      | Notes                                       | 
| ------------------------|:-------------:|:------------------:|:-------------------------------------------:|
| frame	 | util.frame | mirframe
| filterbank | filters.py | mirfilterbank| Librosa-filters.py contains filters for discrete cosine transform, mel frequencies, chroma, logfrequency, constant_q, cq_to_chroma | 
| envelope | -- | mirenvelope
| spectrum |  closest in functionality is core.STFT   | mirspectrum | equivalent to running np.fft.fft
| cepstrum | -- | mircepstrum |  TODO:  Double check; Couldn't find anything that did just cepstrum; equivalent to just np.fft.fft called on fft
| autocor | core.autocorrelate (bounded) | mirautocor
| flux | -- | mirflux
| peaks | core.peak_pick  | mirpeaks | TODO: Librosa uses peak_pick in onset detection; unsure if can be used in isolation
| segment | segment.py |  mirsegment | TODO: Not sure if this is functionally equivalent; both supposedly segment
| sum | -- | mirsum
| rms | feature.rmse | mirrms
| low energy | -- | mirlowenergy
| fluctuation | -- | mirfluctuation
| beat spectrum | beat.beat_track | mirbeatspectrum
| onsets | onset.onset_detect | mironsets
| event density | -- | mireventdensity
| tempo | beat.estimate_tempo | mirtempo
| meter | -- | mirmetre
| metroid | --  | mirmetroid
| pulse clarity | -- | mirpulseclarity
| attack time | -- | mirattacktime
| attack slope | -- | mirattackslope
| attack leap | -- | mirattackleap
| decrease slope | --| mirdecreaseslope
| duration | -- | mirduration | Librosa has core.get_duration but that just returns the full duration of input
| zero cross | feature.zero_crossing_rate | mirzerocross
| roll off | feature.spectral_rolloff | mirrolloff
| brightness | -- | mirbrightness
| centroid | feature.spectral_centroid | mircentroid
| spread  | feature.spectral_spread | mirspread
| skewness | -- | mirskewness
| kurtosis | -- | mirkurtosis
| flatness | -- | mirflatness
| entropy  | -- | mirentropy
| mfcc     | features.mfcc, features.melspectrogram | mirmfcc
| roughness | -- | mirroughness
| regularity | -- | mirregularity
| pitch | -- | mirpitch 
| midi  | core.midi_to_note | mirmidi | other functions in Librosa: core.note_to_midi; midi_to_hz; hz_to_midi
| harmonicity | -- | mirharmonicity 
| chromagram | features.chroma_stft features.chroma_cqt | mirchromagram
| key strength | -- | mirkeystrength | mirkeystrength computes the key strength for each possible key candidate, (cross-correlation of the chromagram)
| key  | -- | mirkey
| mode | -- | mirmode
| keysom | -- | mirkeysom | mirkeysom maps chromagram into a self-organizing map
| tonal centroid | feature.tonnetz | mirtonalcentroid
| hcdf | -- | mirhcdf | hcdf = Harmonic Change Detection Function (flux of tonal centroid)
| similarity matrix | -- | mirsimatrix | In librosa, util.match_intervals matches one set of time intervals to another; not sure how analogous the functionality is though.
| novelty | -- |  mirnovelty
| mean  | -- | mirmean
| std  | -- | mirstd
| median | -- | mirmedian
| stat | -- | mirstat
| histo  | -- | mirhisto | not sure how this would functionally differ from np.histogram
| features | -- | mirfeatures | mirfeatures computes a large set of features, and returns them in a structure array
| map | -- | mirmap
| emotion | -- | miremotion
| classify | -- | mirclassify | Librosa has the pipelining class; uses scikit package for other learning functions
| cluster | -- | mircluster | Librosa has the pipelining class; uses scikit package for clustering
| dist | -- | mirdist
| query | -- | mirquery
| chord | chord.beats_to_chord | --
| harmonic/percussive  comp.| decompose.hpss effects.hpss effects.harmonic effects.percussive | -- | Harmonic percussive source separation
| decompose | decompose.decompose | -- | Decompose spectrogram into components and activations

**** Other librosa functions I couldn't match up
- core.ifgram = instantaneous frequency spectrogram
- effects.pitch_shift, effects.time_stretch, core.phase_vocoder (speed up by a factor of 'rate', given STFT)
- features.pitch_tuning, features.estimate_tuning, features.ifptrack, features.piptrack (Pitch and Tuning features)
- core.A-weighting  = A-weighting of a set of frequencies (unsure of what this means)
- core.perceptual_weighting = perceptual weighting of a power spectrum
- core.localmax = find local max in x (didn't see equivalent in MIR toolbox)
- core.time_to_frames; core.frames_to_time = could be handy in conversion
- util.normalize = normalizing matrices
- util.match_intervals (see note for similarity matrix)
- features.delta (delta features)
- features.stack_memory (short time history embedding. Vertical concatanate a data vector/matrix with delayed copies of itself.)
- features.sync (features of features)
