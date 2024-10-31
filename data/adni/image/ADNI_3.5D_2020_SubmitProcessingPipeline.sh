#!/bin/sh

LIST=$1
MASTERLIST=$2

# Paths
PROJ=/cbica/projects/ADNI/Pipelines/ADNI_3.5D_2020
DATA=${PROJ}/Data/Renamed_Nifti_Data
confFile=${PROJ}/configs/config_3.5D.sh

### Get subject ID and other details from the master csv

#Column	1	PTID_DATE
#Column	5	Phase
#Column	12	IS_BL_Phase
#Column	14	ID_PhaseBaseline


# Get subject from list and corresponding baseline scan
SUB=`awk NR==${SGE_TASK_ID} $LIST`
BLID=`awk -v SUB="$SUB" \
	'BEGIN {FS=","}; \
	{ if ( $1==SUB ) { print $14 } }' \
	$MASTERLIST`

### Print IDs in log file
echo -e "$SGE_TASK_ID \t $SUB \t ${BLID}"

### Run ProcessingPipeline_subject_3.5D
mkdir -p ${PROJ}/Protocols/logs/${SUB}
bash \
 ${PROJ}/Container/singularity_wrapper.sh \
 ${PROJ}/Container/CBICApipeline_centos7.sif \
 ${PROJ}/Scripts/sMRI_ProcessingPipeline/Scripts/ProcessingPipeline_subject_3.5D.sh \
 -ID $SUB \
 -BLID $BLID \
 -T1 ${DATA}/${SUB}/${SUB}_T1.nii.gz \
 -FL ${DATA}/${SUB}/${SUB}_FL.nii.gz \
 -T2 ${DATA}/${SUB}/${SUB}_T2.nii.gz \
 -dest ${PROJ} \
 -MT 4 \
 -config ${confFile} \
 > ${PROJ}/Protocols/logs/${SUB}/ProcessingPipeline_subject_3.5D.sh-${JOB_ID}-${SGE_TASK_ID}.log 2>&1
