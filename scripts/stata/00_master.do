/* -------------------------------------------------------------------------- */
/* FILE: 00_master.do                                                         */
/* DESC: Master script to setup globals and run the analysis pipeline         */
/* -------------------------------------------------------------------------- */

* 1. Initialize Environment
clear all
set more off
macro drop _all
capture log close

* 2. Set Root Path
* (Automatically detects OS)
if c(os) == "MacOSX" {
    global root "/Users/ymw0414/Library/CloudStorage/Dropbox/shifting_slant"
}
else {
    * Windows Path (Your Environment)
    global root "C:/Users/ymw04/Dropbox/shifting_slant"
}

* 3. Define Global Paths (Folder Structure)

* [Input 1] Raw Economic Data (Choi et al. original files)
* Location: data/raw/econ
global raw_data     "$root/data/raw/econ"

* [Input 2] Processed Newspaper Data (Output from Python)
* Location: data/analysis
global py_data      "$root/data/analysis"

* [Output] Processed Stata Datasets (.dta files)
* Location: data/processed/econ
global processed    "$root/data/processed/econ"

* [Alias] Map 'working_data' to 'processed'
* (Critical: This allows your old legacy code to run without changes!)
global working_data "$processed"
global dta          "$processed"

* [Results & Logs]
global results      "$root/results"
global logs         "$root/results/logs"
global scripts      "$root/scripts/stata"
global temp_data    "$root/data/temp"

* 4. Start Logging
* Records everything to 'results/logs/master_log.txt'
log using "$logs/master_log.txt", replace text

display "======================================================================"
display ">>> [00_master] Stata Environment Setup Complete."
display ">>> Root Path:       $root"
display ">>> Raw Data:        $raw_data"
display ">>> Saving Data to:  $processed"
display ">>> Saving Logs to:  $logs"
display "======================================================================"


* 5. Execute Analysis Pipeline

* --- Step 1: Construct Variables ---
* (Replication of Choi et al. Sections 1-5)
* NOTE: Run this ONCE to create .dta files, then comment out to save time.
* do "$scripts/01_build_nafta_vars.do"

* --- Step 2: Merge Data ---
* (Merges Python Newspaper Panel + Stata Economic Data)
do "$scripts/02_import_merge.do"

* --- Step 3: Analysis ---
* (Event Studies & Regressions)
do "$scripts/03_analysis_eventstudy.do"


* 6. Finish
display ">>> All tasks finished successfully."
log close