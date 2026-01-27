#!/bin/bash

function f_calc_free_pct() {
local desired_free=$1

free -b | grep '^Mem' | awk -v x=$desired_free '{ printf("%4.1f \n", (x * 1024 * 1024 * 1024) / $2 * 100 ) }'

}

# Main program

[[ $# -ne 1 ]] && echo "Usage: $0 <integer amount of memory (in gigabytes) to remain free.>" && exit 1

if ( ! type -p bc > /dev/null 2>&1 )
then
	echo "bc(1) is not installed"
	exit 1
fi

[[ ! -w /etc/passwd ]] && echo "You must be superuser to execute $0" && exit 1

LeaveFreeGB=$1
LeaveFreePct=$( f_calc_free_pct $LeaveFreeGB )

echo;echo

echo "Taking action to reduce free memory down to ${LeaveFreeGB}GB available."

sync;sync;sync  # I know modern systems don't need the "3 syncs" but I'm old :).
echo 0 > /proc/sys/vm/nr_hugepages
echo 3 > /proc/sys/vm/drop_caches

# Get total memory in bytes from free -b
TotalMemBytes=$(free -b | awk '/^Mem/ {print $2}')

# Calculate pages to allocate + 2GB buffer
Pages=$(echo "($TotalMemBytes - (($LeaveFreeGB + 2) * 2^30)) / (2 * 2^20)" | bc)

free -g
sync;sync;sync
echo 0 > /proc/sys/vm/nr_hugepages
echo 3 > /proc/sys/vm/drop_caches
echo
echo "Attempting to allocate $Pages huge pages"
echo $Pages > /proc/sys/vm/nr_hugepages


egrep "HugePages_Total|MemAvailable" /proc/meminfo