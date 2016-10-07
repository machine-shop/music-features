#### Interval: 
The distance between any two notes.

 - semitone (half step): The distance between any pitch.

 - wholetone (whole step): Two half steps.


---------------------------


#### Mel scale: 

A perceptual scale of pitches judged by listeners. The lower Mel scales are narrow and indicate how much energy exists near 0 Hz. The higher the frequency the wider is the Mel scale. Already above 500 Hz listeners judge large intervals to have equal pitch increments. [Stevens, Volkman, and Newman, 1937]

  **Implementation steps in librosa:**

  1. The power spectrum within short frames

  2. Apply the mel filterbank and sum the energy in each filter


-----------------------------

#### Root Mean Square

Global energy of the signal; can approximate loudness.  Can be calculated by taking the root average of the square of the amplitude.  

-----------------------------

#### Spectral Centroid

The first moment, or mean, of the signal's spectrum.  It is the geometric center of the distribution. Calculated by taking the sum of the frequencies weighted by the amplitudes, and divided by the sum of the amplitudes. This value is associated with the sound's brightness. 

-----------------------------

#### Spectral Flatness

Indication of how smooth or spiky the distribution is (effectively the noisiness of the spectrum).  Calculated by taking the ratio between the geometric mean of the distribution, and dividing it by the average of the disribution.
Closer to 1, more noiselike, closer to zero, more skipy (i.e. only a couple of pure frequencies are present.)
-----------------------------

#### Spectral Spread

Returns the standard deviation of the spectrum, by calculating the second moment, or variance, of the distribution.  (Standard deviation from the spectral centroid.)  

-----------------------------

#### Zero Crossing Rate

Measures the number of times a signal changes sign in a frame. ZCR is high for noisy/unvoiced signals and low for voiced/tonal signals.  
