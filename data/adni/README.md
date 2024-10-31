# ADNI Dataset Preprocessing
- **To access ADNI dataset, please first visit ADNI website: https://adni.loni.usc.edu/ and apply for data access here: https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=ADNI.**
- Once you get an access, login to IDA and dowlonad necessary files for each modality.
- In essence, for each data modality, please (1) download the neceesary files and (2) run preprocessing code.  
- Please note the date on downloaded file name would be changed based on your downloading date. Thus, update the date in each preprocessing file followingly.

## 0. Diagnosis Data
### Data Folder 
```
data/
├── adni
    └── diagnosis
        └── DXSUM_PDXCONV_22Apr2024.csv
```

### Steps
1. Search & Download -> ARC Builder, Search `DXSUM` and add it to collections.
2. Downloads -> Tables, download `DXSUM`
3. Place the file under `/data/adni/diagnosis/`


## 1. Image (MRI) Preprocessing

### Data Folder 
```
data/
├── adni
    └── image
        └── {Subject ID}_{Scan date}
        └── {Subject ID}_{Scan date}
        └── ...
```

### Steps
1. Search & Download -> Image Collections, Search without Subject ID and save files as `{Subject ID}_{Scan date}` format.
2. Build CBICApipeline container using file: [image/CBICApipeline_centos7.def](image/CBICApipeline_centos7.def)
3. Define singularity wrapper file: [image/singularity_wrapper.sh](image/singularity_wrapper.sh)
4. Define configuration file: [image/config_3.5D.sh](image/config_3.5D.sh)
5. Submit jobs using file: [image/ADNI_3.5D_2020_SubmitProcessingPipeline.sh](image/ADNI_3.5D_2020_SubmitProcessingPipeline.sh)
6. Run python file: [image/GetADNI_Ravens.py](image/GetADNI_Ravens.py)


## 2. Genomic Preprocessing
### Data Folder 
```
data/
├── adni
    └── genomic
        └── ADNI_cluster_01_forward_757LONI.bim
        └── ADNI_cluster_01_forward_757LONI.bed
        └── ADNI_cluster_01_forward_757LONI.fam
        └── ADNI_GO_2_Forward_Bin.bim
        └── ADNI_GO_2_Forward_Bin.bed
        └── ADNI_GO_2_Forward_Bin.fam
        └── ADNI_GO2_GWAS_2nd_orig_BIN.bim
        └── ADNI_GO2_GWAS_2nd_orig_BIN.bed
        └── ADNI_GO2_GWAS_2nd_orig_BIN.fam
        └── ADNI3_PLINK_Final.bim
        └── ADNI3_PLINK_Final.bed
        └── ADNI3_PLINK_Final.fam
        └── ADNI3_PLINK_FINAL_2nd.bim
        └── ADNI3_PLINK_FINAL_2nd.bed
        └── ADNI3_PLINK_FINAL_2nd.fam
```

### Steps
1. Search & Download -> Genetic Files -> Downloads, search below files and download
    - ADNI 1 SNP genotype data - PLINK
    - ADNI GO/2 SNP genotype data - Complete PLINK for sets 1 - 9
    - ADNI GO/2 SNP genotype data - Complete PLINK for sets 10 - 15
    - ADNI3 SNP genotype data Set 1 - PLINK
    - ADNI3 SNP genotype data Set 2 - PLINK
2. Unzip each file and place the unziped files (file name will match with above files) .bim, .bed, .fam files under `/data/adni/genomic/`
3. Move to `data/adni/biospecimen` and follow step-by-step at [genomic/genomic_preprocess.ipynb](genomic/genomic_preprocess.ipynb)


## 3. Clinical Preprocessing
### Data Folder 
```
data/
├── adni
    └── clinical
        └── MEDHIST_09May2024.csv
        └── NEUROEXM_09May2024.csv
        └── PTDEMOG_09May2024.csv
        └── RECCMEDS_09May2024.csv
        └── VITALS_09May2024.csv
```

### Steps
1. Search & Download -> ARC Builder, Search below files, add them into collections and download.
    - MEDHIST
    - NEUROEXM
    - PTDEMOG
    - RECCMEDS
    - VITALS
2. Unzip each file and place the unziped files under `/data/adni/clinical/`
3. Move to `/data/adni/clinical` and run [clinical/clinical_preporcess.py](clinical/clinical_preporcess.py)


## 4. Biospecimen Preprocessing
### Data Folder 
```
data/
├── adni
    └── biospecimen
        └── APOERES_09May2024.csv
        └── UPENNBIOMK_ROCHE_ELECSYS_09May2024.csv
```

### Steps
1. Search & Download -> ARC Builder, Search below files, add them into collections and download.
    - APOERES
    - UPENNBIOMK_ROCHE_ELECSYS
2. Unzip each file and place the unziped files under `/data/adni/biospecimen/`
3. Move to `/data/adni/biospecimen` and run [biospecimen/biospecimen_preprocess.py](biospecimen/biospecimen_preprocess.py)