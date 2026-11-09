#!/bin/bash
# Run by the desktop icon that make_launcher.py makes: launch_player.sh <python> <play_results.py options...>
# Keeps the terminal window open if the player stops with an error, so the message can be read.
cd "$(dirname "$0")" || exit 1
python=$1
shift
if ! "$python" play_results.py "$@"; then
    echo
    read -rp 'The player stopped with an error (see above). Press Enter to close this window.'
fi
