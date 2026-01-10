************************************************************
* 02_merge_newspaper_with_econ.do
* Merge newspaper-level panel with county-year econ data
************************************************************

clear all
cls

* --------------------------------------------------
* Paths
* --------------------------------------------------
global ANALYSIS "C:/Users/ymw04/Dropbox/shifting_slant/data/analysis"
global ECON     "C:/Users/ymw04/Dropbox/shifting_slant/data/processed/econ"

* --------------------------------------------------
* Load newspaper panel (many observations per county-year)
* --------------------------------------------------
import delimited ///
    "$ANALYSIS/newspaper_panel_with_geo.csv", ///
    clear

* Ensure merge keys are numeric and clean
destring year, replace force
destring fips, replace force

rename fips county   // match Stata econ key name

tempfile news
save `news'

* --------------------------------------------------
* Load county-year econ data (unique per county-year)
* --------------------------------------------------
use "$ECON/data_01_build_nafta_vars.dta", clear

destring year county, replace force

* Check uniqueness of county-year in econ data
isid county year

tempfile econ
save `econ'

* --------------------------------------------------
* Merge: newspaper (many) × econ (one)
* --------------------------------------------------
use `news', clear

merge m:1 county year using `econ'

* Keep all newspapers; econ variables missing when not available
drop _merge

* --------------------------------------------------
* Save merged dataset
* --------------------------------------------------
save ///
    "$ANALYSIS/newspaper_panel_with_econ.dta", ///
    replace
