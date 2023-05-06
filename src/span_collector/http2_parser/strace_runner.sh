#!/bin/bash

set +e # otherwise the script will exit on error
elementIn () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

i=0
pids=()
while :
do
  until new_pid=$(pgrep search);
  do
    printf "\nNot yet"
  done
  i=$((i+1))
  elementIn "$new_pid" "${pids[@]}"
  if [[ $? -eq 1 ]]; then
    pids+=( "$new_pid" )
    printf "Running for new pid $new_pid\n"
    sudo strace -f -p $(pgrep search) -v -s 65536 -o output$1-attempt$i.log &
  fi
done
