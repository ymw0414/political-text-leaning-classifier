/* -------------------------------------------------------------------------- */
/* FILE: 03_analysis_eventstudy.do                                            */
/* DESC: Event Study Analysis for Normalized Newspaper Slant                  */
/* - Period: 1988-2004 (Excluding 1986, 1987, 2005)                           */
/* - Base Year: 1993                                                          */
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

display ">>> [Step 03] Starting Analysis (Cleaned Period: 1988-2004)..."

* =========================================================================== *
* 1. Define Program (Optimized for 1988-2004)                                 *
* =========================================================================== *
cap program drop plot_slant_eventstudy
program define plot_slant_eventstudy
    syntax varlist(min=2 max=2), BASEYEAR(integer)

    local depvar : word 1 of `varlist'
    local vulvar : word 2 of `varlist'
    local baseyear = `baseyear'

    * --- Step A: Generate Interaction Terms ---
    quietly levelsof year, local(yrs)
    foreach y of local yrs {
        cap drop _es_`y'
        if `y' != `baseyear' {
            gen _es_`y' = `vulvar' * (year == `y')
        }
    }

    * Manufacturing Controls
    foreach y of local yrs {
        cap drop _manu_`y'
        gen _manu_`y' = manushare1990 * (year == `y')
    }

    local intvars
    local manuvars
    foreach y of local yrs {
        if `y' != `baseyear' {
            local intvars `intvars' _es_`y'
        }
        local manuvars `manuvars' _manu_`y'
    }

    * --- Step B: Run Regressions ---
    * Baseline
    display "   > Running Baseline Model..."
    reghdfe `depvar' `intvars' [aweight = pop1990_total], ///
        absorb(paper year) vce(cluster state)
    
    tempfile est1
    parmest, saving(`est1', replace) format(parm estimate min95 max95)

    * With Controls
    display "   > Running Model with Manufacturing Trends..."
    reghdfe `depvar' `intvars' `manuvars' [aweight = pop1990_total], ///
        absorb(paper year) vce(cluster state)

    tempfile est2
    parmest, saving(`est2', replace) format(parm estimate min95 max95)

    * --- Step C: Process Results ---
    use `est1', clear
    keep if strpos(parm, "_es_") > 0
    gen year = real(substr(parm, 5, .))
    rename (estimate min95 max95) (est1 min1 max1)
    tempfile base
    save `base'

    use `est2', clear
    keep if strpos(parm, "_es_") > 0
    gen year = real(substr(parm, 5, .))
    replace year = year + 0.15
    rename (estimate min95 max95) (est2 min2 max2)

    merge 1:1 year using `base', nogen

    * Add base year (1993)
    set obs `=_N + 1'
    replace year = `baseyear' in L
    replace est1 = 0 in L
    replace min1 = 0 in L
    replace max1 = 0 in L
    
    set obs `=_N + 1'
    replace year = `baseyear' + 0.15 in L
    replace est2 = 0 in L
    replace min2 = 0 in L
    replace max2 = 0 in L

    sort year

    * --- Step D: Plot ---
    twoway ///
        (rcap min1 max1 year, lcolor(midblue%40) lwidth(vthin)) ///
        (scatter est1 year, msymbol(square) mcolor(midblue) mfcolor(white)) ///
        (rcap min2 max2 year, lcolor(orange%40) lwidth(vthin)) ///
        (scatter est2 year, msymbol(diamond) mcolor(orange) mfcolor(white)), ///
        yline(0, lpattern(dot) lcolor(black)) ///
        xline(`baseyear', lpattern(solid) lcolor(gs12)) ///
        ytitle("Effect on Slant (SD units, + = Rep.)") ///
        xtitle("Year") ///
        xlabel(1988(1)2004, labsize(vsmall) angle(45)) ///
        ylabel(, angle(horizontal) labsize(small) nogrid format(%04.2f)) ///
        legend(order(2 "Baseline" 4 "With Manuf. Control") size(small) position(6) rows(1)) ///
        graphregion(color(white)) ///
        title("Impact of NAFTA Exposure on Newspaper Slant") ///
        subtitle("Sample: 1988-2004 (Excl. 86, 87, 05)")
        
end

* =========================================================================== *
* 2. Execution                                                                *
* =========================================================================== *
use "$processed/final_analysis_set.dta", clear

* Filtering
keep if year >= 1988 & year <= 2004

* Run analysis with Base Year 1993
plot_slant_eventstudy slant_normalized vulnerability1990_scaled, baseyear(1993)

graph export "$results/figures/event_study_8804_slant.pdf", replace as(pdf)

display ">>> [Step 03] Analysis Complete for 1988-2004 period."