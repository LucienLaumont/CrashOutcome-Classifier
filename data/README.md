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

## Unexpected values found during audit

After normalizing all years, a value check against the official BAAC codebook revealed the following anomalies:

### characteristics.csv
| Column | Unexpected value | Action |
|--------|-----------------|--------|
| `lum` | `-1` | Replaced with NaN (not in codebook) |
| `int` | `0` or `1` | Replaced with NaN (not in codebook) |

### locations.csv
| Column | Unexpected value | Action |
|--------|-----------------|--------|
| `circ` | `0` | Replaced with NaN (not in codebook) |
| `prof` | `0` | Replaced with NaN (not in codebook) |
| `plan` | `0` | Replaced with NaN (not in codebook) |
| `surf` | `0` | Replaced with NaN (not in codebook) |

### vehicles.csv
| Column | Unexpected value | Action |
|--------|-----------------|--------|
| `catv` | `-1` | Replaced with NaN (only 13 rows across all years, noise) |

### users.csv
| Column | Unexpected value | Action |
|--------|-----------------|--------|
| `catu` | `4` | Deprecated category (roller/scooter). Remapped to `catu=3` (pedestrian) + `catv=99` (other vehicle) at merge time, for consistency with post-2018 data |
| `sexe` | `-1` | Replaced with NaN (not in codebook) |
| `etatp` | `0` | See pedestrian-specific fields below |

### Pedestrian-specific fields (`etatp`, `locp`, `actp`)

These fields only apply to pedestrian rows (`catu=3`). Investigation across all years confirmed:
- **Non-pedestrians** with value `0`: the field is not applicable → replaced with **NaN**
- **Pedestrians** with value `0`: the state is unknown → replaced with **-1** (non renseigné)
- Non-pedestrian rows with values in {1,2,3} (~100 rows total): data entry errors → replaced with **NaN**

### Safety equipment standardization (`secu`)

The encoding of safety equipment changed completely between format eras:

**Pre-2019 (Groups A, B):** Single `secu` column, 2-digit code:
- Tens digit = equipment type: 1=belt, 2=helmet, 3=child device, 4=reflective, 9=other
- Units digit = usage: 1=yes, 2=no, 3=undetermined

**Post-2019 (Group C):** Three columns `secu1`, `secu2`, `secu3`, each with a combined code:
- -1=unknown, 0=none, 1=belt, 2=helmet, 3=child device, 4=reflective vest, 5=airbag, 6=gloves, 7=gloves+airbag, 8=undetermined

**Standardization applied:** Keep only a single `secu1` column representing the equipment type when effectively used:
- Pre-2019: extract equipment type (tens digit) only if usage == 1 (yes), otherwise NaN
- Post-2019: keep `secu1` as-is, drop `secu2` and `secu3` (no multi-equipment info available pre-2019)
- Test set (hybrid): prefer `secu1` if present, otherwise derive from old `secu` with the same logic
- All `-1` values converted to NaN

This means we lose two pieces of information:
1. Whether equipment existed but was **not used** (pre-2019 only) → treated as NaN
2. Second and third equipment (post-2019 only) → dropped for cross-era consistency

### Unified "non renseigné" handling (`-1` and `0` → NaN)

Many columns use `-1` (or `0` in some cases) to encode "non renseigné" (not specified). After audit, all such values have been replaced with NaN for consistency. This allows ML models (LightGBM, XGBoost) to handle them natively as missing values.

| Table | Columns with `-1` → NaN |
|-------|------------------------|
| **characteristics** | `lum`, `atm`, `col`, `int` (also `0` → NaN) |
| **locations** | `circ`, `nbv`, `vosp`, `prof`, `plan`, `surf`, `infra`, `situ` (also `0` → NaN for `circ`, `prof`, `plan`, `surf`) |
| **vehicles** | `catv`, `senc`, `obs`, `obsm`, `choc`, `manv` |
| **users** | `sexe`, `place`, `secu1`, `trajet` (also `0` → NaN) |

**Exceptions** (kept as-is):
- `etatp`, `locp`, `actp` in users: `-1` means "non renseigné" specifically for pedestrians, which is semantically distinct from NaN (field not applicable for non-pedestrians)

### `grav = -1`

Some rows have `grav=-1` (unknown severity). These are dropped entirely since they cannot contribute to the binary target `GRAVE`.

## Other quirks

### Float artifacts (Group B, 2012-2018)
Many integer columns have `.0` suffixes (e.g., `1.0` instead of `1`) due to NaN values causing pandas to cast int columns to float. Affected columns include: `atm`, `col`, `circ`, `nbv`, `vosp`, `prof`, `plan`, `obs`, `obsm`, `choc`, `manv`, `place`, `trajet`, `secu`, `locp`, `actp`, `etatp`, `an_nais`.

### Test set is a hybrid
The test set uses Group A/B formatting (comma separator, 2-digit year, projected coordinates) but includes columns from Group C (`vma`, `id_vehicule`, `motor`, `secu1/2/3`, `id_usager`). It appears to be data from 2012 with an enriched schema.
