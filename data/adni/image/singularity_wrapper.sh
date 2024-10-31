#!/bin/sh

### Timestamps
echo -e "\nRunning commands on          : `hostname`"
echo -e "Start time                   : `date +%F-%H:%M:%S` \n"


echo -e "\n"
set -x

/usr/bin/singularity \
 run \
 -B ${TMPDIR} \
 $*

set +x
echo -e "\n"

