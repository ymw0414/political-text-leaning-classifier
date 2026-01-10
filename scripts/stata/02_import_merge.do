/* -------------------------------------------------------------------------- */
/* FILE: 02_import_merge.do                                                   */
/* DESC: Merges Python Slant Panel with Stata Economic Variables              */
/* - Fixes 'state' type mismatch (String vs Numeric)                          */
/* - Applies State Abbreviation Labels (e.g., 39 -> "OH")                     */
/* -------------------------------------------------------------------------- */

* =========================================================================== *
* [PATH CONFIGURATION]                                                        *
* =========================================================================== *
if "$root" == "" {
    clear all
    set more off
    if c(os) == "MacOSX" {
        global root "/Users/ymw0414/Library/CloudStorage/Dropbox/shifting_slant"
    }
    else {
        global root "C:/Users/ymw04/Dropbox/shifting_slant"
    }
    global raw_data     "$root/data/raw/econ"
    global processed    "$root/data/processed/econ"
}
if "$py_data" == "" {
    global py_data "$root/data/analysis"
}

display ">>> [Step 02] Starting Data Merge..."

* --------------------------------------------------------------------------- *
* 1. Load Python Data & Fix Variables
* --------------------------------------------------------------------------- *
import delimited "$py_data/newspaper_panel_with_geo.csv", clear

display ">>> Loaded Python Data. Fixing State/County formats..."

* --- FIX 1: Rename 'fips' to 'county' ---
capture confirm variable fips
if !_rc {
    rename fips county
}

* --- FIX 2: Ensure 'county' is Numeric ---
capture confirm string variable county
if !_rc {
    destring county, replace force
}

* --- FIX 3: Solve State Conflict (String "OH" vs Numeric 39) ---
* The Python data has 'state' as string (e.g., "OH"). 
* The Econ data has 'state' as numeric FIPS (e.g., 39).
* SOLUTION: Drop the string version and recreate it from 'county' FIPS.

capture drop state 
gen state = floor(county / 1000)
label variable state "State FIPS Code"

* --- FIX 4: Ensure 'year' is clean ---
capture confirm string variable year
if !_rc {
    destring year, replace force
}
else {
    replace year = round(year)
}

sort county year
tempfile text_panel
save `text_panel'


* --------------------------------------------------------------------------- *
* 2. Load Economic Data & Merge
* --------------------------------------------------------------------------- *
capture confirm file "$processed/county_econ_panel.dta"

if _rc == 0 {
    use "$processed/county_econ_panel.dta", clear
    sort county year
    
    display ">>> Merging..."

    * MERGE 1:m
    merge 1:m county year using `text_panel'
    
    * Keep matched
    keep if _merge == 3
    drop _merge

    * ----------------------------------------------------------------------- *
    * 3. Apply State Labels (Numeric -> Abbreviation)
    * ----------------------------------------------------------------------- *
    * Define labels so 39 appears as "OH", 06 as "CA", etc.
    label define state_abbr ///
        1 "AL" 2 "AK" 4 "AZ" 5 "AR" 6 "CA" 8 "CO" 9 "CT" 10 "DE" ///
        11 "DC" 12 "FL" 13 "GA" 15 "HI" 16 "ID" 17 "IL" 18 "IN" 19 "IA" ///
        20 "KS" 21 "KY" 22 "LA" 23 "ME" 24 "MD" 25 "MA" 26 "MI" 27 "MN" ///
        28 "MS" 29 "MO" 30 "MT" 31 "NE" 32 "NV" 33 "NH" 34 "NJ" 35 "NM" ///
        36 "NY" 37 "NC" 38 "ND" 39 "OH" 40 "OK" 41 "OR" 42 "PA" 44 "RI" ///
        45 "SC" 46 "SD" 47 "TN" 48 "TX" 49 "UT" 50 "VT" 51 "VA" 53 "WA" ///
        54 "WV" 55 "WI" 56 "WY"
    
    label values state state_abbr
    
    * Label other variables
    label variable slant_weighted "Slant (Weighted)"
    label variable vulnerability1990_scaled "NAFTA Vulnerability"
    label variable manushare1990 "Manufacturing Share 1990"

    * Save Final Dataset
    save "$processed/final_analysis_set.dta", replace
    
    display "=========================================================="
    display ">>> [Step 02] Complete!"
    display ">>> Saved: $processed/final_analysis_set.dta"
    display ">>> Variable 'state' is now Numeric with Labels (e.g. 39->OH)."
    display "=========================================================="
}
else {
    display as error ">>> [ERROR] Economic data file not found!"
    exit 198
}