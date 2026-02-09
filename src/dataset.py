from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_DIR = _PROJECT_ROOT / "data" / "train"
_TEST_DIR = _PROJECT_ROOT / "data" / "test"

# ──────────────────────────────────────────────
# Canonical column sets (output order)
# ──────────────────────────────────────────────

CANONICAL_CHARACTERISTICS = [
    "Num_Acc", "an", "mois", "jour", "hrmn",
    "lum", "agg", "int", "atm", "col", "com", "dep",
]

CANONICAL_LOCATIONS = [
    "Num_Acc", "catr", "voie", "v1", "v2",
    "circ", "nbv", "pr", "pr1", "vosp", "prof", "plan",
    "lartpc", "larrout", "surf", "infra", "situ",
]

CANONICAL_VEHICLES = [
    "Num_Acc", "num_veh", "senc", "catv", "occutc",
    "obs", "obsm", "choc", "manv",
]

CANONICAL_USERS = [
    "Num_Acc", "num_veh", "place", "catu", "grav",
    "sexe", "an_nais", "trajet",
    "secu1", "secu2", "secu3",
    "locp", "actp", "etatp",
]

VALUES_MAP_CHARACTERISTICS = {
    "lum": [1,2,3,4,5], "agg": [1,2], "int": [1,2,3,4,5,6,7,8,9], "atm": [-1,1,2,3,4,5,6,7,8,9],
    "col": [-1,1,2,3,4,5,6,7],
}

VALUES_MAP_LOCATIONS = {
    "catr": [1,2,3,4,5,6,7,8,9], "circ": [-1,1,2,3,4],
    "vosp": [-1,0,1,2,3], "prof": [-1,1,2,3,4], "plan": [-1,1,2,3,4],
    "surf": [-1,1,2,3,4,5,6,7,8,9], "infra": [-1,0,1,2,3,4,5,6,7,8,9],
    "situ": [-1,0,1,2,3,4,5,6,7,8],
}

VALUES_MAP_VEHICLES = {
    "senc": [-1,0,1,2,3], "catv": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,30,31,32,33,34,35,36,37,38,39,40,41,42,43,50,60,80,99], 
    "obs": [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], "obsm": [-1,0,1,2,4,5,6,9],
    "choc": [-1,0,1,2,3,4,5,6,7,8,9], "manv": [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
    "motor": [-1,0,1,2,3,4,5,6]
}

VALUES_MAP_USERS = {
    "catu": [1,2,3], "sexe": [1,2], "trajet": [-1,0,1,2,3,4,5,9], 
    "secu1": [-1,0,1,2,3,4,5,6,7,8,9], "secu2": [-1,0,1,2,3,4,5,6,7,8,9], "secu3": [-1,0,1,2,3,4,5,6,7,8,9],
    "actp": [-1,0,1,2,3,4,5,6,7,8,9,"A","B"], "etatp": [-1,1,2,3]
}

# ──────────────────────────────────────────────
# Target dtypes per table (nullable ints for NaN support)
# Columns not listed here keep their inferred dtype.
# ──────────────────────────────────────────────

_DTYPES_CHARACTERISTICS = {
    "Num_Acc": "Int64",
    "an": "Int16", "mois": "Int8", "jour": "Int8", "hrmn": "Int16",
    "lum": "Int8", "agg": "Int8", "int": "Int8",
    "atm": "Int8", "col": "Int8",
}

_DTYPES_LOCATIONS = {
    "Num_Acc": "Int64",
    "catr": "Int8",
    "circ": "Int8", "nbv": "Int8",
    "vosp": "Int8", "prof": "Int8", "plan": "Int8",
    "surf": "Int8", "infra": "Int8", "situ": "Int8",
}

_DTYPES_VEHICLES = {
    "Num_Acc": "Int64",
    "senc": "Int8", "catv": "Int16", "occutc": "Int16",
    "obs": "Int8", "obsm": "Int8", "choc": "Int8", "manv": "Int8",
}

_DTYPES_USERS = {
    "Num_Acc": "Int64",
    "place": "Int8", "catu": "Int8", "grav": "Int8",
    "sexe": "Int8", "an_nais": "Int16", "trajet": "Int8",
    "secu1": "Int8", "secu2": "Int8", "secu3": "Int8",
    "locp": "Int8", "actp": "Int8", "etatp": "Int8",
}


def _cast_dtypes(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """Cast columns to nullable integer types."""
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return df


# ──────────────────────────────────────────────
# Format group helpers
# ──────────────────────────────────────────────

def _resolve_group(year: Union[int, str]) -> str:
    if year == "test":
        return "D"
    year = int(year)
    if year in (2010, 2011):
        return "A"
    if 2012 <= year <= 2018:
        return "B"
    if 2019 <= year <= 2022:
        return "C"
    raise ValueError(f"Unsupported year: {year}")


def _csv_params(group: str) -> dict:
    if group in ("A", "D"):
        return dict(sep=",", encoding="latin-1", low_memory=False)
    return dict(sep=";", index_col=0, encoding="latin-1", low_memory=False)


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────

class Dataset:
    """Load and normalize one year (or test set) of BAAC data.

    Usage::

        ds = Dataset(2015)
        ds.characteristics   # normalized DataFrame
        ds.merged()          # all 4 tables joined

        ds_test = Dataset("test")
    """

    def __init__(self, year: Union[int, str]) -> None:
        self.year = year
        self.group: str = _resolve_group(year)
        self._data_dir: Path = (
            _TEST_DIR if year == "test"
            else _TRAIN_DIR / f"BAAC-Annee-{year}"
        )
        self._characteristics: pd.DataFrame | None = None
        self._locations: pd.DataFrame | None = None
        self._vehicles: pd.DataFrame | None = None
        self._users: pd.DataFrame | None = None

    # ── Properties (lazy load) ───────────────

    @property
    def characteristics(self) -> pd.DataFrame:
        if self._characteristics is None:
            self._characteristics = self._norm_characteristics(self._read("characteristics"))
        return self._characteristics

    @property
    def locations(self) -> pd.DataFrame:
        if self._locations is None:
            self._locations = self._norm_locations(self._read("locations"))
        return self._locations

    @property
    def vehicles(self) -> pd.DataFrame:
        if self._vehicles is None:
            self._vehicles = self._norm_vehicles(self._read("vehicles"))
        return self._vehicles

    @property
    def users(self) -> pd.DataFrame:
        if self._users is None:
            self._users = self._norm_users(self._read("users"))
        return self._users

    # ── Merge ────────────────────────────────

    def merged(self) -> pd.DataFrame:
        """Join all 4 tables into a single DataFrame."""
        df = self.characteristics.merge(self.locations, on="Num_Acc", how="inner")
        df = df.merge(self.vehicles, on="Num_Acc", how="inner")
        df = df.merge(self.users, on=["Num_Acc", "num_veh"], how="inner")

        # Remap deprecated catu=4 (roller/scooter) → catu=3 (pedestrian) + catv=99
        mask_catu4 = df["catu"] == 4
        df.loc[mask_catu4, "catv"] = 99
        df.loc[mask_catu4, "catu"] = 3

        return df

    # ── Raw read ─────────────────────────────

    def _read(self, table: str) -> pd.DataFrame:
        path = self._data_dir / f"{table}.csv"
        return pd.read_csv(path, **_csv_params(self.group))

    # ── Normalize: characteristics ───────────

    def _norm_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Fix float artifacts (Group B / D)
        for col in ("atm", "col"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalize an → 4-digit
        if self.group in ("A", "B", "D"):
            df["an"] = df["an"].apply(lambda x: x + 2000 if x < 100 else x)

        # Normalize hrmn → integer HHMM
        if self.group == "C":
            df["hrmn"] = (
                df["hrmn"]
                .astype(str)
                .str.replace(":", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

        # Replace unexpected values with NaN
        df["lum"] = df["lum"].replace(-1, pd.NA)
        df["int"] = df["int"].replace({0: pd.NA, -1: pd.NA})

        # Keep com/dep as strings
        df["com"] = df["com"].astype(str)
        df["dep"] = df["dep"].astype(str)

        # Drop + reorder to canonical + cast dtypes
        df = df.reindex(columns=CANONICAL_CHARACTERISTICS)
        return _cast_dtypes(df, _DTYPES_CHARACTERISTICS)

    # ── Normalize: locations ─────────────────

    def _norm_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Fix float artifacts
        for col in ("circ", "nbv", "vosp", "prof", "plan", "surf", "infra", "situ"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace unexpected 0s with NaN
        for col in ("circ", "prof", "plan", "surf"):
            if col in df.columns:
                df[col] = df[col].replace(0, pd.NA)

        # Keep string columns as strings
        for col in ("pr", "pr1", "lartpc", "larrout", "voie", "v1", "v2"):
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", np.nan)

        df = df.reindex(columns=CANONICAL_LOCATIONS)
        return _cast_dtypes(df, _DTYPES_LOCATIONS)

    # ── Normalize: vehicles ──────────────────

    def _norm_vehicles(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Fix float artifacts
        for col in ("senc", "obs", "obsm", "choc", "manv"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace catv=-1 with NaN (noise, only 13 rows)
        df["catv"] = df["catv"].replace(-1, pd.NA)

        df = df.reindex(columns=CANONICAL_VEHICLES)
        return _cast_dtypes(df, _DTYPES_VEHICLES)

    # ── Normalize: users ─────────────────────

    def _norm_users(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Fix float artifacts
        for col in ("place", "trajet", "locp", "actp", "etatp", "an_nais"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle secu → secu1/secu2/secu3
        if self.group in ("A", "B"):
            secu = pd.to_numeric(df["secu"], errors="coerce")
            df["secu1"] = secu // 10
            df["secu2"] = secu % 10
            df["secu3"] = -1
            df = df.drop(columns=["secu"])

        elif self.group == "D":
            # Test set hybrid: both secu and secu1/2/3 exist
            secu = pd.to_numeric(df.get("secu"), errors="coerce")
            for c in ("secu1", "secu2", "secu3"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if secu is not None and "secu1" in df.columns:
                mask = df["secu1"].isna() & secu.notna()
                df.loc[mask, "secu1"] = secu[mask] // 10
                df.loc[mask, "secu2"] = secu[mask] % 10
                df.loc[mask, "secu3"] = -1
            df = df.drop(columns=["secu"], errors="ignore")

        # Group C: secu1/2/3 already present, nothing to do

        # Replace sexe=-1 with NaN
        df["sexe"] = df["sexe"].replace(-1, pd.NA)

        # Pedestrian-specific fields (etatp, locp, actp)
        is_ped = df["catu"] == 3
        for col in ("etatp", "locp", "actp"):
            if col in df.columns:
                # Non-pedestrians: not applicable → NaN
                df.loc[~is_ped, col] = pd.NA
                # Pedestrians with 0: not specified → -1
                df.loc[is_ped & (df[col] == 0), col] = -1

        # Ensure grav column exists (absent in test)
        if "grav" not in df.columns:
            df["grav"] = pd.array([pd.NA] * len(df), dtype="Int8")

        # Drop rows where grav == -1 (unknown severity, unusable for training)
        df = df[df["grav"].isna() | (df["grav"] != -1)]

        df = df.reindex(columns=CANONICAL_USERS)
        return _cast_dtypes(df, _DTYPES_USERS)

    # ── Value checking ──────────────────────

    def check_values(self) -> dict[str, dict[str, set]]:
        """Return unexpected values for each table.

        Returns a dict like::

            {
                "characteristics": {"col": {8, 99}},
                "locations": {},
                ...
            }

        An empty inner dict means all values are valid.
        """
        table_map = {
            "characteristics": (self.characteristics, VALUES_MAP_CHARACTERISTICS),
            "locations":       (self.locations,       VALUES_MAP_LOCATIONS),
            "vehicles":        (self.vehicles,        VALUES_MAP_VEHICLES),
            "users":           (self.users,           VALUES_MAP_USERS),
        }
        result = {}
        for table_name, (df, values_map) in table_map.items():
            unexpected = {}
            for col, valid in values_map.items():
                if col not in df.columns:
                    continue
                actual = set(df[col].dropna().unique())
                invalid = actual - set(valid)
                if invalid:
                    unexpected[col] = invalid
            result[table_name] = unexpected
        return result

    # ── Repr ─────────────────────────────────

    def __repr__(self) -> str:
        return f"Dataset(year={self.year!r}, group={self.group!r})"
