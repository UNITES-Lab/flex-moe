#/bin/sh

############################################################
### Decide which steps in the processing pipeline to execute
############################################################

RunLPS="Yes"
RunN4="Yes"
RunSS="Yes"
RunT2ICV="Yes"
RunFASTBC="Yes"
RunMUSE="Yes"
RunMUSEfeats="Yes"
RunSEG="Yes"
RunRAVENS="Yes"
RunRAVENSProc="Yes"


############################################################
### Set parameters for each step in the pipeline
############################################################

### Reorientation
if [ $RunLPS == "Yes" ]
then
	# path
	LPS=${dest}/Protocols/ReOrientedLPS
	
	# suffixes
	T1LPSsuffix="${T1suffix}_LPS"
	T2LPSsuffix="${T2suffix}_LPS"
	FLLPSsuffix="${FLsuffix}_LPS"
	PDLPSsuffix="${PDsuffix}_LPS"
else
	# suffixes
	T1LPSsuffix="${T1suffix}"
	T2LPSsuffix="${T2suffix}"
	FLLPSsuffix="${FLsuffix}"
	PDLPSsuffix="${PDsuffix}"
fi

### N4 bias correction
if [ $RunN4 == "Yes" ]
then
	# path
	N4=${dest}/Protocols/BiasCorrected
	
	# parameters
	N4iter=1
	
	# suffixes
	T1N4suffix="${T1LPSsuffix}_N4"
	T2N4suffix="${T2LPSsuffix}_N4"
	FLN4suffix="${FLLPSsuffix}_N4"
	PDN4suffix="${PDLPSsuffix}_N4"
else
	# suffixes
	T1N4suffix="${T1LPSsuffix}"
	T2N4suffix="${T2LPSsuffix}"
	FLN4suffix="${FLLPSsuffix}"
	PDN4suffix="${PDLPSsuffix}"
fi



### Skull Stripping
if [ $RunSS == "Yes" ]
then
	# path
	SS=${dest}/Protocols/Skull-Stripped

	### MUSE
	# parameters
	MuseSSTempNum=50
	MuseSSTempLoc=${MUSE_DIR}/data/Templates/BrainExtraction
	MuseSSrois=${MUSE_DIR}/data/List/MUSE-SS_ROIs.csv
	MuseSSMethod=3
	MuseSSTemps=15
	MuseSSDRAMMSReg=0.1
	MuseSSANTSReg=0.5
	
	# suffixes
	SSMaskSuffix="${T1N4suffix}_brainmask_muse-ss"
	SSBrainSuffix="${T1N4suffix}_brain_muse-ss"
	
	SSMeanMaskSuffix="${SSMaskSuffix}_Mean"
	SSMeanBrainSuffix="${SSBrainSuffix}_Mean"
	
	FinalSSMasksuffix="${SSMeanMaskSuffix}"
	FinalSSsuffix="${SSMeanBrainSuffix}"
fi

### T2 ICV
if [ $RunT2ICV == "Yes" ]
then
	# parameters
	T2ICViter=100
	T2ICVminsd=-1
	T2ICVmaxsd=100
	T2ICVtol=0.00001
	
	# suffixes
	T2ICVsuffix="${FinalSSMasksuffix}_ICV"
fi

### FAST bias correction
if [ $RunFASTBC == "Yes" ]
then
	# path
	FASTBC=${dest}/Protocols/fastbc
	
	# parameters
	FASTiter=8
	FASTfwhm=20
	
	# suffixes
	T1fastbcsuffix="${FinalSSsuffix}_fastbc"
fi

### MUSE
if [ $RunMUSE == "Yes" ]
then
	# path
	MUSE=${dest}/Protocols/MUSE
	
	# parameters
	MuseCSF=1.2
	MUSE_ROI_TOOLS=${software}/MUSE_ROI_TOOLS
	MuseInt=0
	MuseDRAMMSReg=0.1
	MuseANTSReg=0.5
	MuseMethod=3
	MuseTemps=11
	MuseTempPath=${MUSE_DIR}/data/Templates/WithCere
	MuseTempNum=35
	
	case $MuseMethod in
		1)
			MUSEfilesuffix="muse";
			MUSEmethodslist="dramms";;
		2)
			MUSEfilesuffix="muse";
			MUSEmethodslist="ants";;
		3)
			MUSEfilesuffix="muse";
			MUSEmethodslist="dramms ants";;
	esac

	# suffixes
	MUSEsuffix="${T1fastbcsuffix}_${MUSEfilesuffix}"
fi

### Segmentation
if [ $RunSEG == "Yes" ]
then
	# path
	SEG=${dest}/Protocols/Segmented
	
	# suffixes
	MUSESEGsuffix="${MUSEsuffix}_seg"
fi

### RAVENS
if [ $RunRAVENS == "Yes" ]
then
	# path
	RAVENS=${dest}/Protocols/RAVENS
	
	# parameters
	RavensTemplate=${templates}/RAVENS/BLSA_SPGR+MPRAGE_averagetemplate.nii.gz
	RavensTemplatebName=`basename ${RavensTemplate%.nii.gz}`
#	RavensReg=0.5
#	RavensMethod=ants
	RavensReg=0.3
	RavensMethod=dramms
	RavensNorm=1750000
	RavensDef="Delete"
	RavensScaleFactor=1000
	
	# suffixes
	RAVENSsuffix="${MUSESEGsuffix}_${RavensMethod}-${RavensReg}_RAVENS"
fi

### RAVENS post processing
if [ $RunRAVENSProc == "Yes" ]
then
	# path
	Norm=${dest}/Protocols/RAVENS
	Smooth=${dest}/Protocols/RAVENS
	DS=${dest}/Protocols/RAVENS
fi
