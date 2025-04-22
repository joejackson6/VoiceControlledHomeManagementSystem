#!/bin/bash

INPUT_DIR="${1:-$(pwd)/VoiceData}"
LOG_FILE="conversion_errors.log"
> "$LOG_FILE"

echo "Using input directory: $INPUT_DIR"
echo "Logging errors to: $LOG_FILE"

find "$INPUT_DIR" -type f -name "*.m4a" | while read -r file; do
	dir=$(dirname "$file")
	base=$(basename "$file" .m4a)
	wav_file="$dir/${base}.wav"

	if [ -f "$wav_file" ]; then
		echo "Already exists: $wav_file"
		continue
	fi

	echo "Converting: $file → $wav_file"
	ffmpeg -y -i "$file" -ac 1 -ar 16000 "$wav_file" 2>> "$LOG_FILE"

	if [ ! -f "$wav_file" ]; then
		echo "FAILED to convert: $file" >> "$LOG_FILE"
	fi
done

echo "Conversion complete. See $LOG_FILE for any failures."
