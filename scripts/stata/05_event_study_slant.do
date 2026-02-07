/* ------------------------------------------------------------------------- */
/* FILE: 05_event_study_slant.do                                             */
/* DATE: 2026-01-11                                                          */
/* DESC: Event Study Analysis of NAFTA Impact on Newspaper Slant              */
/*       (Widmer-style binary outcome)                                       */
/* ------------------------------------------------------------------------- */

clear all
set more off
cls

* --------------------------------------------------
* 1. Set Directory Paths
* --------------------------------------------------
global root "C:/Users/ymw04/Dropbox/shifting_slant"
global analysis "$root/data/analysis"
global results  "$root/results"

capture mkdir "$results"

* --------------------------------------------------
* 2. Load Data
* --------------------------------------------------
display ">>> Loading final dataset..."
use "$analysis/data_04_final_dataset.dta", clear

* Newspaper FE id
egen news_id = group(paper)

* Drop low-coverage newspaper-years
drop if n_articles < 100

* --------------------------------------------------
* 3. Define Event Study Program
* --------------------------------------------------
cap program drop my_eventstudy2
program define my_eventstudy2
    syntax varlist(min=2 max=2), BASEYEAR(integer) position(integer)

    local depvar : word 1 of `varlist'
    local vulvar : word 2 of `varlist'
    local baseyear = `baseyear'
    local position = `position'

    display ">>> Outcome: `depvar' | Shock: `vulvar' | Base Year: `baseyear'"

    * --- Interaction terms ---
    quietly levelsof year, local(yrs)
    local intvars
    foreach y of local yrs {
        if `y' != `baseyear' {
            cap drop _es_`y'
            gen _es_`y' = `vulvar' * (year == `y')
            local intvars `intvars' _es_`y'
        }
    }

    * --- Regression: Newspaper FE + Year FE only ---
    reghdfe `depvar' `intvars', ///
        absorb(news_id year) vce(robust)

    tempfile est
    parmest, saving(`est', replace) format(parm estimate min95 max95)

    * --- Prepare plot data ---
    use `est', clear
    keep if strpos(parm, "_es_") > 0
    gen year = real(substr(parm, 5, .))
    rename estimate est
    rename min95 min
    rename max95 max

    * Add base year = 0
    set obs `=_N + 1'
    replace year = `baseyear' in L
    replace est = 0 in L
    replace min = 0 in L
    replace max = 0 in L

    sort year

    * --- Plot ---
    sum year
    local xmin = r(min)
    local xmax = r(max)

    twoway ///
        (rcap min max year, lcolor(gs9)) ///
        (scatter est year, msymbol(square) mcolor(black) mfcolor(white)), ///
        yline(0, lpattern(dot) lcolor(black)) ///
        xline(`baseyear', lpattern(dash) lcolor(gs10)) ///
        ytitle("Share of Republican Articles") ///
        xtitle("Year") ///
        title("Event Study: NAFTA Exposure and Newspaper Slant") ///
        xlabel(`xmin'(2)`xmax', labsize(small) angle(45)) ///
        ylabel(, angle(horizontal) nogrid) ///
        legend(off) ///
        graphregion(color(white)) ///
        plotregion(style(none))
end

* --------------------------------------------------
* 4. Run Event Study
* --------------------------------------------------
display ">>> Running Event Study..."

* Outcome: Widmer binary slant
* Shock: vulnerability1990_scaled
* Base year: 1993
my_eventstudy2 slant_share_rep vulnerability1990_scaled, ///
    baseyear(1993) position(7)

* --------------------------------------------------
* 5. Save Figure
* --------------------------------------------------
graph export "$results/figure_event_study_slant_gray.pdf", replace

display "=========================================================="
display ">>> SUCCESS: Event Study Completed"
display ">>> Graph saved to: $results/figure_event_study_slant_gray.pdf"
display "=========================================================="
