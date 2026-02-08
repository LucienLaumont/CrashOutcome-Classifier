# Data Inconsistencies Report

The BAAC dataset spans 2010-2022 but the CSV format changed multiple times. This document lists every inconsistency to be aware of before loading and merging the data.

## Format groups

The 13 training years fall into **3 distinct format eras**, and the test set is a 4th hybrid format.

| Group | Years | Separator | Index column | `read_csv` |
|-------|-------|-----------|-------------|------------|
| **A** | 2010-2011 | `,` | No | `pd.read_csv(path, sep=',', encoding='latin-1')` |
| **B** | 2012-2018 | `;` | Yes (unnamed, 0-based) | `pd.read_csv(path, sep=';', index_col=0, encoding='latin-1')` |
| **C** | 2019-2022 | `;` | Yes (unnamed, 0-based) | `pd.read_csv(path, sep=';', index_col=0, encoding='latin-1')` |
| **D** | test | `,` | No | `pd.read_csv(path, sep=',', encoding='latin-1')` |

## characteristics.csv

### Year format (`an`)
| Group | Format | Example |
|-------|--------|---------|
| A, B, D | 2-digit | `10`, `18`, `12` |
| C | 4-digit | `2019`, `2022` |

### Time format (`hrmn`)
| Group | Format | Example |
|-------|--------|---------|
| A, B, D | Integer HHMM | `1930`, `615` |
| C | String HH:MM | `08:35`, `02:50` |

### Coordinates (`lat`, `long`)
| Group | Decimal separator | Coordinate system | Example |
|-------|-------------------|-------------------|---------|
| A | N/A (integers) | Projected (Lambert?) | `5051600`, `0292000` |
| B, D | `.` (dot) | Projected (Lambert?) | `5052928.0`, `293643.0` |
| C | `,` (comma) | WGS84 decimal degrees | `48,6900000`, `2,4100000` |

**This is critical**: Groups A/B/D use projected coordinates, Group C uses standard GPS coordinates. They cannot be mixed without conversion.

### Column changes
| Column | A (2010-2011) | B (2012-2018) | C (2019-2022) | D (test) |
|--------|---------------|---------------|---------------|----------|
| `gps` | Present | Present | **Removed** | Present |
| `dep` format | 3-digit (`590`) | 3-digit (`590`) | 2-digit (`91`) | 3-digit |
| `com` format | 3-digit commune | 3-digit commune | Full INSEE (`91657`) | 3-digit |
| Column order | `an, mois, jour` | `an, mois, jour` | **`jour, mois, an`** | `an, mois, jour` |

## locations.csv

| Column | A, B (2010-2018) | C (2019-2022) | D (test) |
|--------|-------------------|---------------|----------|
| `env1` | Present | **Removed** | Present |
| `vma` (speed limit) | Absent | **Added** | Present |
| Column order of `vosp`, `prof`, `pr`, `pr1` | Original order | **Reshuffled** | Original |

## vehicles.csv

| Column | A, B (2010-2018) | C (2019-2022) | D (test) |
|--------|-------------------|---------------|----------|
| `id_vehicule` | Absent | **Added** | Present (appended) |
| `motor` | Absent | **Added** | Present (appended) |

## users.csv

| Column | A, B (2010-2018) | C 2019-2020 | C 2021-2022 | D (test) |
|--------|-------------------|-------------|-------------|----------|
| `secu` | Single column | **Split into `secu1`, `secu2`, `secu3`** | Split | Split (+ original `secu`) |
| `id_vehicule` | Absent | Added | Added | Present |
| `id_usager` | Absent | Absent | **Added** | Present |
| `grav` | Present | Present | Present | **Absent** (target) |

## Other quirks

### Float artifacts (Group B, 2012-2018)
Many integer columns have `.0` suffixes (e.g., `1.0` instead of `1`) due to NaN values causing pandas to cast int columns to float. Affected columns include: `atm`, `col`, `circ`, `nbv`, `vosp`, `prof`, `plan`, `obs`, `obsm`, `choc`, `manv`, `place`, `trajet`, `secu`, `locp`, `actp`, `etatp`, `an_nais`.

### Test set is a hybrid
The test set uses Group A/B formatting (comma separator, 2-digit year, projected coordinates) but includes columns from Group C (`vma`, `id_vehicule`, `motor`, `secu1/2/3`, `id_usager`). It appears to be data from 2012 with an enriched schema.
