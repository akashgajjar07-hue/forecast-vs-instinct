"""Demand Intelligence outputs (EDA). Runs sql/*.sql through DuckDB via src/eda.py.

Writes
  outputs/eda/demand.json            -> matches SCHEMAS['demand.json'] (validated here)
  outputs/eda/ecuador_analogue.json  -> list matching SCHEMAS['scenario.json'].ecuador_analogue (validated here)
  outputs/eda/eda_extras.json        -> payday, earthquake, transactions, oil, PRODUCE gap, analogue detail

History used: 2015-06-01 .. last calibration target date (2017-06-27). Holdout untouched.
"""
import json
import sys
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import eda  # noqa: E402
from src.schemas import SCHEMAS  # noqa: E402

OUT = C.OUTPUTS / "eda"


def dump(obj, path: Path, compact=True):
    txt = json.dumps(obj, separators=(",", ":")) if compact else json.dumps(obj, indent=1)
    path.write_text(txt)
    return len(txt.encode())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = eda.connect()
    demand = eda.build_demand(con)
    jsonschema.validate(demand, SCHEMAS["demand.json"])
    analogue, d0, dm1 = eda.build_analogue(con)
    jsonschema.validate(analogue, SCHEMAS["scenario.json"]["properties"]["ecuador_analogue"])
    extras = eda.build_extras(con, d0, dm1)

    size = dump(demand, OUT / "demand.json")
    assert size < 400_000, f"demand.json too large: {size} bytes"
    dump(analogue, OUT / "ecuador_analogue.json", compact=False)
    dump(extras, OUT / "eda_extras.json", compact=False)
    print(f"demand.json {size/1024:.1f} KB, weeks={len(demand['weekly']['week_start'])}, "
          f"series={len(demand['volatility'])}; cutoff={eda.EDA_CUTOFF}")
    for a in analogue:
        print(f"  analogue {a['family']:<14} {a['change_pct']:+7.2f}%  n_holidays={a['n_holidays']}")


if __name__ == "__main__":
    main()
