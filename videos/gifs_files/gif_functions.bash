# Gif-related
# ===========

function gifspeedchange() {
	# args: $gif_path $frame_delay (1 = 0.1s)
        local orig_gif="${1?'Missing GIF filename parameter'}"
        local frame_delay=${2?'Missing frame delay parameter'}
	gifsicle --batch --delay $frame_delay $orig_gif
	local newframerate=$(echo "$frame_delay*10" | bc)
	echo "new GIF frame rate: $newframerate ms"
}

function gifopt() {
	# args: $input_file ($loss_level)
	if [ -z "$2" ]
	then
		# use default of 30
		local loss_level=30
	elif [[ "$2" =~ ^[0-9]+$ ]] && [ "$2" -ge 30 -a "$2" -le 200 ]
	then
        echo "using loss_level $2"
		local loss_level=$2
	else
		echo "${2:-"Loss level parameter must be an integer from 30-200"}" 1>&2
		#exit 1
	fi
    	if [ -z "$3" ]
	then
		# use default of 256
		local color_level=256
	elif [[ "$3" =~ ^[0-9]+$ ]] && [ "$3" -ge 4 -a "$3" -le 256 ]
	then
        echo "using color level $3"
		local color_level=$3
	else
		echo "${3:-"Color parameter must be an integer from 4-256"}" 1>&2
		#exit 1
	fi
    	if [ -z "$4" ]
	then
		# use default of 256
		local scale=1.0
	elif [[ "$4" =~ ^[+-]?[0-9]+\.?[0-9]*$ ]]
	then
		echo "Using scale $4"
        local scale=$4
	else
		echo "${4:-"Scale from 0.1-4.0"}" 1>&2
		#exit 1
	fi
	local inputgif="${1?'Missing input file parameter'}"
	local gifname="$(basename $inputgif .gif)"
	local basegifname=$(echo "$gifname" | sed 's/_reduced_x[0-9]//g')
	local outputgif="$basegifname-optim.gif"
	gifsicle -O3 $gifdelay --lossy="$loss_level" --colors "$color_level" --scale "$scale" -o "$outputgif" "$inputgif";
	local oldfilesize=$(du -h $inputgif | cut -f1)
	local newfilesize=$(du -h $outputgif | cut -f1)
	echo "File reduced from $oldfilesize to $newfilesize as $outputgif"
}

function gif_framecount_reducer () {
	# args: $gif_path $frames_reduction_factor
	local orig_gif="${1?'Missing GIF filename parameter'}"
	local reduction_factor=${2?'Missing reduction factor parameter'}
	# Extracting the delays between each frames
	local orig_delay=$(gifsicle -I "$orig_gif" | sed -ne 's/.*delay \([0-9.]\+\)s/\1/p' | uniq)
	# Ensuring this delay is constant
	[ $(echo "$orig_delay" | wc -l) -ne 1 ] \
		&& echo "Input GIF doesn't have a fixed framerate" >&2 \
		&& return 1
	# Computing the current and new FPS
	local new_fps=$(echo "(1/$orig_delay)/$reduction_factor" | bc)
	# Exploding the animation into individual images in /var/tmp
	local tmp_frames_prefix="/var/tmp/${orig_gif%.*}_"
	convert "$orig_gif" -coalesce +adjoin "$tmp_frames_prefix%05d.gif"
	local frames_count=$(ls "$tmp_frames_prefix"*.gif | wc -l)
	# Creating a symlink for one frame every $reduction_factor
	local sel_frames_prefix="/var/tmp/sel_${orig_gif%.*}_"
	for i in $(seq 0 $reduction_factor $((frames_count-1))); do
		local suffix=$(printf "%05d.gif" $i)
		ln -s "$tmp_frames_prefix$suffix" "$sel_frames_prefix$suffix"
	done
	# Assembling the new animated GIF from the selected frames
	convert -delay $new_fps "$sel_frames_prefix"*.gif "${orig_gif%.*}_reduced_x${reduction_factor}.gif"
	# Cleaning up
	rm "$tmp_frames_prefix"*.gif "$sel_frames_prefix"*.gif
}
