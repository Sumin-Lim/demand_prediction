#!/bin/zsh
intervals=('30min' '1H' '2H' '4H' '6H' '8H' '12H' '24H')
slidings=(4 6 8 12 24)

for interval in "${intervals[@]}"; do
  for sliding in "${slidings[@]}"; do
    python dae.py -i "$interval" -n "$sliding"
  done
done
