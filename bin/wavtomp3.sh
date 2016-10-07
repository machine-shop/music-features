#!/bin/bash
# Converts wav files to mp3 files using LAME encoder

SAVEIFS=$IFS   # Save the current field separator 
IFS=$(echo -en "\n\b") # Overwrite to deal with filenames with spaces

FILES="/auto/k2/stimuli/music/smd/*.wav"
#for file in $(ls *wav)
for file in $FILES
do
  name=${file%%.wav}
  lame -V0 -h -b 192 --vbr-new $name.wav $name.mp3
done

IFS=$SAVEIFS # Restore IFS env variable back to its original value
