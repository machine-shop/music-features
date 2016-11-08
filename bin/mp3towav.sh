#!/bin/bash
# Converts mp3 files to wav files using mpg321 encoder

SAVEIFS=$IFS   # Save the current field separator 
IFS=$(echo -en "\n\b") # Overwrite to deal with filenames with spaces

FILES="/auto/k2/stimuli/music/smd/*.mp3"
#for file in $(ls *mp3)
for file in $FILES
do
  name=${file%%.mp3}
  #lame -V0 -h -b 192 --vbr-new $name.wav $name.mp3  # wav to mp3
  mpg321 -w $name.wav $name.mp3
done

IFS=$SAVEIFS # Restore IFS env variable back to its original value
