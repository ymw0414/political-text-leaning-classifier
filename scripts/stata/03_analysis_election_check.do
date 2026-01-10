/* -------------------------------------------------------------------------- */
/* FILE: 03_analysis_election_check.do                                        */
/* DESC: Sanity check using Election Results as the Dependent Variable        */
/* To see if NAFTA shocks actually translated into political behavior   */
/* MODS: 1. Identify election variables / 2. Run Event Study on GOP share     */
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
    global processed "$root/data/processed/econ"
    global results   "$root/results"
}

* =========================================================================== *
* 1. Data Loading & Variable Identification                                   *
* =========================================================================== *
use "$processed/final_analysis_set.dta", clear

* Search for election-related variables (Assuming names like gop_share or rep_vote)
lookfor election vote gop rep dem

* NOTE: Presidential elections happen in 1988, 1992, 1996, 2000, 2004.
* We will filter for these years if using Presidential vote shares.
keep if inlist(year, 1988, 1992, 1996, 2000, 2004)

* =========================================================================== *
* 2. Regression: NAFTA Shock on Voting Behavior                               *
* =========================================================================== *
* Let's assume the variable name is 'gop_share'. Replace if it's different.
local dv "gop_share"
local baseyear = 1992

* --- Generate Interactions ---
foreach y in 1988 1996 2000 2004 {
    gen _vote_es_`y' = vulnerability1990_scaled * (year == `y')
}

display ">>> Running Sanity Check: Effect on GOP Vote Share..."

* Regression with County (or Paper) FE and Year FE
* Using County FE is more standard for election data
reghdfe `dv' _vote_es_* [aweight = pop1990_total], absorb(fips year) vce(cluster state)

* =========================================================================== *
* 3. Plotting the Result                                                      *
* =========================================================================== *
tempfile est_vote
parmest, saving(`est_vote', replace) format(parm estimate min95 max95)

use `est_vote', clear
keep if strpos(parm, "_vote_es_") > 0
gen year = real(substr(parm, 10, .))

* Add base year (1992)
set obs `=_N + 1'
replace year = `baseyear' in L
replace estimate = 0 in L
replace min95 = 0 in L
replace max95 = 0 in L
sort year

twoway ///
    (rcap min95 max95 year, lcolor(cranberry)) ///
    (scatter estimate year, msymbol(circle) mcolor(cranberry) mfcolor(white)), ///
    yline(0, lpattern(dot) lcolor(black)) ///
    xline(`baseyear', lpattern(solid) lcolor(gs12)) ///
    ytitle("Effect on GOP Vote Share") ///
    xtitle("Election Year") ///
    title("Sanity Check: NAFTA Exposure and Voting Behavior") ///
    subtitle("Outcome: Republican Vote Share (Base: 1992)") ///
    graphregion(color(white))

graph export "$results/figures/sanity_check_election.pdf", replace as(pdf)

display ">>> [Step 03] Sanity Check Complete."