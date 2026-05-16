# Holacratic MATPERIW-MCDM
# Prof. Dr. Adil Baykasoglu © May 16, 2026

This software that is developed by Adil Baykasoglu is capable of **solving hierarchical MCDM problems**. 
It determines **subjective criterion weights** using the **Analytic Hierarchy Process (AHP)** and 
**objective criterion weights** using the **Entropy method**. 
These are integrated within the **matrix permanent** structure **without requiring any additional processing**.

Based on the **input file structure**, once the problem is defined, 
the system **automatically handles the entire workflow** and **ranks the alternatives** accordingly.

## Project layout

- MATPERIW_HOLACRATIC.py — core holacratic MATPERIW-MCDM implementation
- RUN_matperiw_MCDM_holacratic.py — example runner (Excel input → results + export)
- USER_MANUAL.md — workbook format, API, dependencies, and troubleshooting

## Quick start

Install dependencies (see `USER_MANUAL` for the full list), prepare an `.xlsx` input as described there, then run:

```bash
python RUN_matperiw_MCDM_holacratic.py
```

---

*Implementation: see source header in `MATPERIW_HOLACRATIC.py` for authorship and date.*
