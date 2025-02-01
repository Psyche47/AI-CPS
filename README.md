## This repository is a part of the course **‘M. Grum: Advanced AI-based Application Systems’**.

### Project Team Members:

- Emmanuel Barshan Gomes
- Md Ataur Rahman

### Dataset used: Air Quality Dataset UCI

> Dataset Link: (https://archive.ics.uci.edu/dataset/360/air+quality)

The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level,within an Italian city. Data were recorded from March 2004 to February 2005 (one year)representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities as well as both concept and sensor drifts are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 (citation required) eventually affecting sensors concentration estimation capabilities. Missing values are tagged with -200 value.
This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.

### Variable Table

| Variable Name | Type        | Description                                                                                             | Units      |
| ------------- | ----------- | ------------------------------------------------------------------------------------------------------- | ---------- |
| Date          | Date        |                                                                                                         |            |
| Time          | Categorical |                                                                                                         |            |
| CO(GT)        | Integer     | True hourly averaged concentration CO in mg/m^3 (reference analyzer)                                    | mg/m^3     |
| PT08.S1(CO)   | Categorical | Hourly averaged sensor response (nominally CO targeted)                                                 |            |
| NMHC(GT)      | Integer     | True hourly averaged overall Non-Methanic HydroCarbons concentration in microg/m^3 (reference analyzer) | microg/m^3 |
| C6H6(GT)      | Continuous  | True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)                           | microg/m^3 |
| PT08.S2(NMHC) | Categorical | Hourly averaged sensor response (nominally NMHC targeted)                                               |            |
| NOx(GT)       | Integer     | True hourly averaged NOx concentration in ppb (reference analyzer)                                      | ppb        |
| PT08.S3(NOx)  | Categorical | Hourly averaged sensor response (nominally NOx targeted)                                                |            |
| NO2(GT)       | Integer     | True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)                               | microg/m^3 |
| PT08.S4(NO2)  | Categorical | Hourly averaged sensor response (nominally NO2 targeted)                                                |            |
| PT08.S5(O3)   | Categorical | Hourly averaged sensor response (nominally O3 targeted)                                                 |            |
| T             | Continuous  | Temperature                                                                                             | °C         |
| RH            | Continuous  | Relative Humidity                                                                                       | %          |
| AH            | Continuous  | Absolute Humidity                                                                                       |            |
