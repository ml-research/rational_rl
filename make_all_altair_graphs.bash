#!/bin/bash
# Read a string with spaces using for loop
for value in Asterix BattleZone Breakout Enduro Jamesbond Kangaroo Pong Qbert Seaquest Skiing SpaceInvaders Tennis TimePilot Tutankham VideoPinball
do
    python3 altair_vizu.py $value
done
