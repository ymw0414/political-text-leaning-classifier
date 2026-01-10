************************************************************
*            Section 0: Directory Setup & Define Program   *
************************************************************

* ---------- Directory ---------- *
clear all
cls 

if c(os) == "MacOSX" {
    global mydirectory "/Users/ymw0414/Library/CloudStorage/Dropbox/shifting_slant"
}
else {
    global mydirectory "C:/Users/ymw04/Dropbox/shifting_slant"
}

global raw_data     "$mydirectory/data/raw/econ"
global working_data "$mydirectory/data/processed/econ"
global results      "$mydirectory/results"



***********************************************************
*            Section 1: Ad Valorem Equivalent Tariff      *
***********************************************************

* ---------- 1989-2001 tariffs from Romalis ---------- *
foreach y in 90 91 92 93 94 95 96 97 98 99 00 01 {
    import delimited "$raw_data/tariff/tariff_89-01_romalis/USHTS`y'.TXT", clear
	replace year = `y' if missing(year)
	keep hts8 year brief_description quantity_1_code mexico_ad_val_rate mexico_specific_rate unitvalue
	save "$working_data/tariff_romalis_`y'.dta", replace
}

clear

foreach y in 90 91 92 93 94 95 96 97 98 99 00 01 {
	append using "$working_data/tariff_romalis_`y'.dta"
}

replace year = year + 1900 if year < 100 // Convert 2-digit years to 4-digit (e.g., 90 → 1990); skip if already 4-digit


save "$working_data/tariff_romalis_all", replace 

* ---------- 2002-2008 tariffs from USITC ---------- *
forvalues y = 2002/2008 {
	import delimited "$raw_data/tariff/tariff_02-17_usitc/tariff_database_`y'.txt", clear
	gen year = `y'
	keep hts8 year brief_description quantity_1_code mexico_rate_type_code mexico_ad_val_rate mexico_specific_rate
	capture confirm string variable hts8
		if !_rc {
			drop if real(hts8) == .
			destring hts8, replace 
		}
	save "$working_data/tariff_usitc_`y'.dta", replace 
}

clear

forvalues y = 2002/2008 {
	append using "$working_data/tariff_usitc_`y'.dta"
}

save "$working_data/tariff_usitc_all", replace

* ---------- Merging unitvalue ---------- *
import excel ///
    "$raw_data/import/import_usitc/dataweb-queryExport.xlsx", ///
    sheet("Customs Value") ///
    cellrange(A3:W11599) ///
    firstrow ///
    clear
	
reshape long Year, i(HTSNumber QuantityDescription) j(year)
rename Year value

save "$working_data/value", replace

import excel ///
    "$raw_data/import/import_usitc/dataweb-queryExport.xlsx", ///
    sheet("First Unit of Quantity") ///
    cellrange(A3:W11599) ///
    firstrow ///
    clear
	
reshape long Year, i(HTSNumber QuantityDescription) j(year)
rename Year quantity

merge 1:1 HTSNumber QuantityDescription year using "$working_data/value"

keep if year >= 2002
rename HTSNumber hts8

gen unitvalue = value/quantity // (141,044 missing values generated)

gen unit_abbrev = ""
replace unit_abbrev = "KG" if QuantityDescription == "kilograms"
replace unit_abbrev = "L" if QuantityDescription == "liters"
replace unit_abbrev = "X" if QuantityDescription == "cubic meters"
replace unit_abbrev = "T" if QuantityDescription == "metric tons"
replace unit_abbrev = "M2" if QuantityDescription == "square meters"
replace unit_abbrev = "PCS" if QuantityDescription == "pieces"
replace unit_abbrev = "NA" if QuantityDescription == "no units collected"
replace unit_abbrev = "PFL" if QuantityDescription == "proof liters"
replace unit_abbrev = "NO" if QuantityDescription == "number"
replace unit_abbrev = "THS" if QuantityDescription == "thousand units"
replace unit_abbrev = "M3" if QuantityDescription == "thousands of cubic meters"
replace unit_abbrev = "KM3" if QuantityDescription == "thousand meters"
replace unit_abbrev = "BBL" if QuantityDescription == "barrels"
replace unit_abbrev = "MWH" if QuantityDescription == "megawatt hours"
replace unit_abbrev = "GM" if QuantityDescription == "gold content grams"
replace unit_abbrev = "G" if QuantityDescription == "grams"
replace unit_abbrev = "DOZ" if QuantityDescription == "dozens"
replace unit_abbrev = "THM" if QuantityDescription == "ton raw value"
replace unit_abbrev = "GCN" if QuantityDescription == "component grams"
replace unit_abbrev = "M" if QuantityDescription == "meters"
replace unit_abbrev = "DPR" if QuantityDescription == "dozen pairs"
replace unit_abbrev = "GRS" if QuantityDescription == "gross"
replace unit_abbrev = "CYK" if QuantityDescription == "clean yield kilograms"
replace unit_abbrev = "PRS" if QuantityDescription == "pairs"
replace unit_abbrev = "SQ" if QuantityDescription == "squares"
replace unit_abbrev = "DPC" if QuantityDescription == "dozen pieces"
replace unit_abbrev = "CTN" if QuantityDescription == "component tons"
replace unit_abbrev = "CM2" if QuantityDescription == "square centimeters"
replace unit_abbrev = "GR" if QuantityDescription == "component kilograms"
replace unit_abbrev = "HUN" if QuantityDescription == "hundred units"
replace unit_abbrev = "CAR" if QuantityDescription == "carats"
replace unit_abbrev = "SME" if QuantityDescription == "fiber meters"
replace unit_abbrev = "PK" if QuantityDescription == "pack"
replace unit_abbrev = "LNM" if QuantityDescription == "linear meters"
replace unit_abbrev = "FBM" if QuantityDescription == "Megabecquerels"
replace unit_abbrev = "DS" if QuantityDescription == "doses"
replace unit_abbrev = "CGM" if QuantityDescription == "component tons"
replace unit_abbrev = "CKG" if QuantityDescription == "clean yield kilograms"
replace unit_abbrev = "KTS" if QuantityDescription == "tons"

gen quantity_1_code = unit_abbrev

sort year hts8
duplicates report year hts8
keep year hts8 unitvalue quantity_1_code
replace hts8 = subinstr(hts8, ".", "", .)  
destring hts8, replace  

save "$working_data/unitvalue_1990-2008.dta", replace

use "$working_data/tariff_usitc_all", clear

merge m:1 hts8 year quantity_1_code using "$working_data/unitvalue_1990-2008.dta"
keep if _merge == 1 | _merge == 3
drop _merge

destring mexico_rate_type_code, replace force

save "$working_data/tariff_usitc_all_unitvalue", replace

* ---------- Calculating ad valorem equivalent tariff rate ---------- *

use "$working_data/tariff_romalis_all", clear

append using "$working_data/tariff_usitc_all_unitvalue"

replace mexico_ad_val_rate = . if mexico_ad_val_rate == 10000
replace mexico_specific_rate = . if mexico_specific_rate == 10000

gen tariff_AVE = mexico_ad_val_rate + (mexico_specific_rate / unitvalue)

replace tariff_AVE = mexico_ad_val_rate if missing(unitvalue) & mexico_specific_rate == 0

keep hts8 year tariff_AVE
order year hts8 tariff_AVE

bysort year hts8: egen tariff_AVE_max = max(tariff_AVE) // Processing duplicates with 'max'
drop tariff_AVE
duplicates drop (year hts8), force
rename tariff_AVE_max tariff_AVE

save "$working_data/tariff_all", replace


***********************************************************
*            Section 2: Weights and RCA                   * 
***********************************************************
// Note: Generate weights using imports data (1990) and revealed comparative advantage from exports (1990)

* ---------- Imports ---------- *
use "$raw_data/import/import_hakobyan_mclaren/imports1990-2000", clear

keep if year == 1990
drop year

save "$working_data/import_1990", replace

use "$working_data/tariff_all", clear

merge m:1 hts8 using "$working_data/import_1990"
keep if _merge == 3
drop _merge

gen hts6 = floor(hts8 / 100)
rename hts6 hs6 

save "$working_data/tariff_all_import", replace

* ---------- Exports ---------- *
import delimited ///
    "$raw_data/export/export_hakobyan_mclaren/worldex19902000.csv", ///
    clear
	
keep if year == 1990
drop year tradeflowcode

replace mexin1000usd = 0 if mexin1000usd == .
replace usain1000usd = 0 if usain1000usd == .
gen wldex = wldin1000usd - (mexin1000usd + usain1000usd)

drop mexin1000usd usain1000usd wldin1000usd

reshape wide wldex, i(productcode) j(reporteriso3) string
replace wldexMEX = 0 if  wldexMEX == .

rename productcode hs6 // productcode = hs6 

gen wldexROW = wldexAll - wldexMEX

drop if hs6 == "9999AA"
destring hs6, replace

save "$working_data/export_1990", replace

use "$working_data/tariff_all_import", clear

merge m:1 hs6 using "$working_data/export_1990"
keep if _merge == 3
drop _merge

save "$working_data/tariff_all_import_export", replace

* ---------- Convert hs6 to sic ---------- *
use "$raw_data/etc/crosswalk_david_dorn/cw_hs6_sic87dd", clear

rename sic87dd sic
drop if missing(sic)

duplicates tag (sic hs6), gen(duplicates)
tab duplicates
duplicates drop (sic hs6), force
replace share = 1 if duplicates == 1 // Different from the original paper

drop weights_method duplicates

save "$working_data/hs6_sic", replace

use "$working_data/tariff_all_import_export", clear

joinby hs6 using "$working_data/hs6_sic"

* ---------- Generating weigthed tariff ---------- *

duplicates tag (year sic hs6), gen(duplicates)

foreach ex in wldexAll wldexMEX wldexROW {
	replace `ex' = `ex'/ (duplicates +1)
}

foreach i in tot_im mex_im wldexAll wldexMEX wldexROW {
	replace `i' = `i' * share
}

bysort sic year: egen tot_mex_im = total(mex_im)
gen weight_im = mex_im / tot_mex_im // This part is not introduced in the paper, but the weights were used in their replication codes.

gen tariff_AVE_wtd = tariff_AVE * weight_im

* ---------- Generating "revealed comparative advantage (RCA)" ---------- *

bysort year sic: egen wldexMEX_sictot = total(wldexMEX)
bysort year sic: egen wldexROW_sictot = total(wldexROW)

keep year sic tariff_AVE_wtd tariff_AVE wldexMEX_sictot wldexROW_sictot 

duplicates drop year sic, force 

bysort year: egen wldexMEX_tot = total(wldexMEX_sictot)
bysort year: egen wldexROW_tot = total(wldexROW_sictot)

gen rca_MEX = (wldexMEX_sictot/wldexROW_sictot)/(wldexMEX_tot/wldexROW_tot)

keep year sic tariff_AVE_wtd tariff_AVE rca_MEX

drop if missing(rca_MEX)

gen temp1 = rca_MEX if year == 1990
bysort sic: egen temp2 = max(temp1)
gen rca1990_MEX = temp2
drop temp*

save "$working_data/tariff_rca_MEX", replace


***********************************************************
*            Section 3: Employment                        *
***********************************************************

foreach year in 86 87 88 89 90 91 92 93 94 95 96 97 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
    import delimited "$raw_data/cbp/cbp`year'co.txt", clear
    gen year = `year'
	replace year = cond(year <= 97, year + 1900, year)
    save "$working_data/cbp`year'co.dta", replace
}

clear

foreach year in 86 87 88 89 90 91 92 93 94 95 96 97 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
    append using "$working_data/cbp`year'co.dta"
}

drop if fipscty == 999
gen county = fipstate * 1000 + fipscty

gen emp_tot = emp if sic == "----" | naics == "------" // 1997 and before 1997 → SIC, after → NAICS
bysort year county: egen temp = max(emp_tot) 
replace emp_tot = temp

* ---------- Assigning median employment values for each range ---------- *
replace emp = 10 if empflag == "A"
replace emp = 60 if empflag == "B"
replace emp = 175 if empflag == "C"
replace emp = 375 if empflag == "E"
replace emp = 750 if empflag == "F"
replace emp = 1750 if empflag == "G"
replace emp = 3750 if empflag == "H"
replace emp = 7500 if empflag == "I"
replace emp = 17500 if empflag == "J"
replace emp = 37500 if empflag == "K"
replace emp = 75000 if empflag == "L"
egen above100000_med = median(emp) if emp >= 100000
replace emp = above100000_med if empflag == "M"

destring sic naics, replace force
drop if (missing(sic) & year <= 1997) | (missing(naics) & year > 1997)

* ---------- Generating manufacturing dummy ---------- * 
gen manufacturing = (inrange(sic, 2000, 3999)| inrange(naics, 310000, 339999)) if !missing(sic) | !missing(naics)

keep fipstate fipscty sic emp year naics county manufacturing

save "$working_data/employment", replace


***********************************************************
*            Section 4: Vulnerability Measure             *
***********************************************************

use "$working_data/employment", clear // There is no missing data in the 'year' variable.

keep if year == 1990
keep fipstate county sic emp
rename emp emp1990

joinby sic using "$working_data/tariff_rca_MEX", unmatched(master) // There are missing values in the 'year' variable since some counties in the master data do not match with the using data, i.e., there are missing tariff values in some counties. For those counties, the 'year' variable has missing values since the 'year' variable comes from the using data.

bysort county year: egen numerator = total(emp * rca_MEX * tariff_AVE_wtd)
bysort county year: egen denominator = total(emp * rca_MEX)

gen vulnerability = numerator/denominator
gen temp = vulnerability if year == 1990 
bysort county: egen vulnerability1990 = mode(temp)

duplicates drop county year, force

keep fipstate county year vulnerability* 

rename fipstate state

drop if inlist(state, 2,15) // Restricting to the contiguous 48 states and DC (exclude Alaska = 2, Hawaii = 15)

preserve
	keep if year == 1990
	xtile quartile = vulnerability1990, n(4)
	keep county quartile

	save "$working_data/quartile", replace
restore

merge m:1 county using "$working_data/quartile"
drop _merge

preserve
	keep if year == 1990
	summarize vulnerability1990 if quartile == 4, meanonly
	local top_mean = r(mean)
	summarize vulnerability1990 if quartile == 1, meanonly
	local bottom_mean = r(mean)
	gen vulnerability1990_scaled = vulnerability1990 / (`top_mean' - `bottom_mean')
	
	save "$working_data/vulnerability_scaled", replace
restore

merge m:1 county using "$working_data/vulnerability_scaled"
drop _merge

save "$working_data/vulnerability", replace


***********************************************************
*            Section 5: Pop./Edu./Manu. Share/Income      *
***********************************************************

* ---------- Population: 1986-1989 ---------- *
forvalues year = 1986/1989 {
    import excel using "$raw_data/demographic/county_by_sex_race_age_8089.xls", ///
        sheet("`year'") cellrange(A6:U18853) firstrow clear
    drop in 1 // empty rows
	
	rename (YearofEstimate FIPSStateandCountyCodes) (year county)
	gen state = substr(county, 1, 2)
	
	egen pop_total = rowtotal(Under5years-yearsandover), missing 
	egen pop_working = rowtotal(to19years-to64years), missing 
	
	collapse (sum) pop_*, by(state county year) // Although `county` contains state info, I include `state' for readability.
	
	keep year county state pop_total pop_working 

	save "$working_data/pop`year'.dta", replace
}

clear 

forvalues year = 1986/1989 {
    append using "$working_data/pop`year'.dta"
}

save "$working_data/pop_80s", replace

* ---------- Population: 1990-1999 ---------- *
forvalues year = 1990/1999 {
	use "$raw_data/demographic/stch`year'.dta", clear
	
	bysort state county: egen pop_total = total(pop) // egen total() ignores missing values by default, so no ,missing option is needed.
	bysort state county: egen pop_working = total(pop) if inrange(agegroup, 4, 13)	// Age 15-64
	bysort state county: egen pop_white = total(pop) if inlist(racesex, 1, 2)

	collapse (firstnm) year pop_*, by(state county)

	save "$working_data/pop`year'.dta", replace
}

clear 

forvalues year = 1990/1999 {
    append using "$working_data/pop`year'.dta"
}

save "$working_data/pop_90s", replace

* ---------- Population: 2000-2008 ---------- *
use "$raw_data/demographic/coest00intalldata.dta", clear

drop if inlist(yearref, 1, 12) // Remove observations where yearref equals 1 (April 1, 2000) or 12 (April 1, 2010) since the others are all July 1 data.

keep if year >= 2000 & year <= 2008
drop if agegrp == 99 // Total number

bysort year county: egen pop_total = total(tot_pop)
bysort year county: egen pop_working  = total(tot_pop) if inrange(agegrp, 4, 13)

collapse (firstnm) state pop_total pop_working, by(year county)

save "$working_data/pop_00s", replace

clear 

foreach file in pop_80s pop_90s pop_00s {
    append using "$working_data/`file'.dta"
} 

destring state county, replace

gen temp = pop_total if year == 1990
bysort county: egen pop1990_total = max(temp)
drop temp

save "$working_data/pop_all", replace 

* ---------- Education ---------- *
import excel using "$raw_data/education/Education.xls", ///
    sheet("Education 1970 to 2017") cellrange(A5:AV3667) firstrow clear 
	
drop if FIPSCode == "" // There are missing values at the end of the data. 

rename (FIPSCode AB AC AD Percentofadultswithabachelo) (county less_highschool1990 highschool1990 college1990 bachelor_higher1990) 
keep (county less_highschool1990 highschool1990 college1990 bachelor_higher1990) // Keeping variables in 1990
drop if substr(county, -3, 3) == "000" // Drop county-level aggregates. These are rows where the FIPS code ends in "000".
gen state = substr(county, 1, 2)
destring state county, replace

replace county = 12025 if county == 12086 // Correcting Dade -> Miami-Dade
replace county = 46113 if county == 46102 // Correcting Shannon -> Oglala Lakota

foreach var in less_highschool1990 highschool1990 college1990 bachelor_higher1990 {
    replace `var' = `var' / 100
    local oldlabel : variable label `var'
    local newlabel : subinstr local oldlabel "percent" "", all
    label variable `var' "`newlabel'"
} // Converting percentage variables to proportions and removes the word "percent" from their variable labels. 

save "$working_data/education1990", replace

* ---------- Manufacturing Share ---------- *
import delimited "$raw_data/cbp/cbp90co.txt", clear

keep if sic == "----" | sic == "20--" // sic == "20--" is a CBP summary code that gives total manufacturing employment (SIC 20–39).
drop if fipscty == 999

gen county = fipstate * 1000 + fipscty

keep fipstate county sic emp

replace sic = "total" if sic == "----"
replace sic = "manufacturing" if sic == "20--"

reshape wide emp, i(fipstate county) j(sic) string

gen manushare1990 = empmanufacturing / emptotal
gen year = 1990

rename (fipstate emp*) (state emp_*1990)

keep state county manushare1990 emp_*

save "$working_data/manushare1990", replace

* ---------- Income ---------- * 
// This part mimics the original paper's replication code a bit due to its complexity, but I slightly changed and added detailed explanations for each step.
import delimited "$raw_data/income/median_hh_income.csv", clear rowrange(10:3260)
drop if missing(v1) | strpos(v1, ",") == 0 // Dropping missing values and non-county rows such as state-level aggregates
split v1, parse(",") // ex) v1: Arlington County, VA -> v11: Arlington County variable, v12: VA.
destring v2 v3 v4 v5, ignore(",") force replace 
keep v2 v3 v4 v5 v12 v11
rename (v2 v3 v4 v5 v12 v11) ///
	(county_income_1999 county_income_1989 county_income_1979 county_income_1969 st_abrv county_name)
replace st_abrv = trim(st_abrv) // Removes leading and trailing spaces.
replace county_name = subinstr(county_name, "County", "", .) // Removes the word "County" from the variable county_name.
replace county_name = subinstr(county_name, "Census Area", "", .)
replace county_name = subinstr(county_name, "Parish", "", .)
replace county_name = subinstr(county_name, "Borough", "", .)
replace county_name = subinstr(county_name, "city", "City", .) if st == "VA"
replace county_name = subinstr(county_name, ".", "", .)
replace county_name = subinstr(county_name, "\'", "", .)
replace county_name = "Yellowstone Nat Park" if county_name == "Yellowstone National Park"
replace county_name = lower(ustrtrim(county_name))
replace county_name = "la plata" if county_name == "laplata"
replace county_name = "prince georges" if county_name == "prince george's"
replace county_name = "queen annes" if county_name == "queen anne's"
replace county_name = "st marys" if county_name == "st mary's"
replace county_name = "colonial heights cit" if county_name=="colonial heights city" & st_abr=="VA"
replace county_name = "de baca" if county_name == "debaca" & st_abr=="NM"
replace county_name = "de kalb" if county_name == "dekalb" & inlist(st_abr, "AL", "GA", "IL", "MO", "TN")
replace county_name = "de soto" if county_name == "desoto" & inlist(st_abr, "FL", "MS")
replace county_name = "de witt" if county_name == "dewitt" & st_abr=="TX"
replace county_name = "du page" if county_name == "dupage" & st_abr=="IL"
replace county_name = "la grange" if county_name == "lagrange" & st_abr=="IN"
replace county_name = "la moure" if county_name == "lamoure" & st_abr=="ND"
replace county_name = "mckean" if county_name == "mc kean" & st_abr=="PA"
replace county_name = "o brien" if county_name == "o'brien" & st_abr=="IA"
replace county_name = "ste. genevieve" if county_name == "ste genevieve" & st_abr=="MO"

duplicates drop county_name st_abrv, force

preserve

	use "$raw_data/crosswalk/countyfip_name_crosswalk", clear
	
	replace county_name = lower(ustrtrim(county_name)) // 'ustrtrim' removes Unicode whitespace — including non-breaking spaces, tabs, and other special space characters.
	
	save "$working_data/countyfip_name_crosswalk_lower", replace
	
restore

merge 1:1 county_name st_abrv using "$working_data/countyfip_name_crosswalk_lower" 
// Mapping county_name to FIPS code using crosswalk data. 
// Not match (from master): 4.  These counties are historical or very small and do not match modern FIPS crosswalks).
keep if _merge == 3
drop _merge

rename (statefip countyfips county_income_*) (state county income*)

keep state county income1989

save "$working_data/income", replace

* ---------- Merging employment ---------- *
use "$working_data/employment", clear

rename fipstate state

keep year state county emp

bysort year county: egen temp = total(emp)
replace emp = temp 
drop temp
duplicates drop year county, force

save "$working_data/employment_county", replace

* ---------- Merging population, education, manufacturing share, income, employment ---------- *
use "$working_data/vulnerability", clear

merge 1:1 year county using "$working_data/pop_all"
keep if inlist(_merge, 1, 3)
drop _merge

merge m:1 county using "$working_data/education1990"
keep if inlist(_merge, 1, 3)
drop _merge

merge m:1 county using "$working_data/manushare1990"
keep if inlist(_merge, 1, 3)
drop _merge

merge m:1 county using "$working_data/income"
keep if inlist(_merge, 1, 3)
drop _merge

merge 1:1 year county using "$working_data/employment_county"
drop _merge

sort state county year

keep state county year vulnerability1990_scaled manushare1990 emp pop1990_total

foreach var in vulnerability1990_scaled manushare1990 pop1990_total {
    bysort county: egen temp = max(`var')
    replace `var' = temp
    drop temp
}

xtset county year
xtbalance, range(1986 2008) miss(_all)

save "$working_data/data_01_build_nafta_vars", replace 






















