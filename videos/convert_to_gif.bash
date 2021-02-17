#!/bin/bash
for filename in *.mp4; do
  outputfile=$(echo $filename | sed "s/.mp4/.gif/")
  if [ -f "$outputfile" ]; then
      printf "\033[0;31m $outputfile already exists. \033[0m\n"
    else
      printf "\033[0;32m converting $filename to $outputfile. \033[0m\n"
      ffmpeg -i $filename -vf "fps=10,scale=512:-1:flags=lanczos" -c:v pam -f image2pipe - | convert -delay 5 - -loop 0 -layers optimize $outputfile
  fi
done
