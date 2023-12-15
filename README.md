# GVS and risk-taking - A replication of De Maio et al., 2021

All scripts are provided as R Markdown and HTML files to ensure transparency, reproducibility, and easy understanding of the code. 
The following scripts are part of this repository:

- **A_Counterbalancing**: This script ensures the counterbalancing of the 3 stimulation blocks for both experiments. It also contains the stimulation block information that will be later joined with the session number information in the B2_processing file.
- **B1_Processing**: This script simply converts the .csv output files from PsychoPy to .tsv files. We will use .tsv files to adhere to the BIDS standards (https://bids.neuroimaging.io/).
- **B2_Processing**: This script contains relevant data wrangling and tidying steps for the behavioral data. Furthermore, it applies the prespecified exclusion criteria and extracts and forms the variables relevant for the later analyses.
- **C_Analyses**: This script contains all preregistered analyses.

Note that the repository also includes dummy data that has been generated by us. This data was merely used to optimally prepare the code for the real data and to ensure easy examinability of the code by others. The real data will be added after data acquisition. 
The current folders contain the following information/files:

- **conditions**: This folder contains the .csv output file of the *A_Counterbalancing* script, listing the required stimulation block for each session
- **data_test**: This folder contains the dummy data stored according to BIDS standards (note that the required .json files are still missing, but will be added with the real data).
- **processed**: This folder contains the processed .csv files for both experiments from the *B2_Processing* script.
- **raw**: This folder contains the raw data created by PsychoPy. The script *B1_Processing* converts these .csv files to .tsv files in the same folder. The converted files can then be stored in the data folder according to BIDS standards.
