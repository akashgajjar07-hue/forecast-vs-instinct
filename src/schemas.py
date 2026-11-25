"""JSON Schemas for every file the dashboard reads (docs/data/*.json).
Human-readable version: docs/data_contract.md. CI validates docs/data against these."""
from __future__ import annotations

NUM = {"type": "number"}
NUMN = {"type": ["number", "null"]}
STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}
DATE = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}
DATES = {"type": "array", "items": DATE}
FAM_ENUM = {"type": "string", "enum": ["BEVERAGES", "DAIRY", "BREAD/BAKERY", "GROCERY I", "FROZEN FOODS",
                                       "CLEANING", "PERSONAL CARE", "PRODUCE"]}
MODEL_ENUM = {"type": "string", "enum": ["naive", "seasonal_naive", "moving_average", "holt_winters", "lightgbm"]}
BAND_ENUM = {"type": "string", "enum": ["Accept", "Monitor", "Review"]}
# Per-series values are published as an index (value / private series base x 100, src/indexing.py), never in units.
# Field names that would carry raw competition data are forbidden everywhere (tests/test_no_raw_data.py).
FORBIDDEN_FIELDS = ["sales", "actual", "onpromotion", "promo_total", "mean_daily_sales", "mean_promo_count",
                    "max_promo_count", "forecast_14d", "p10_14d", "p90_14d", "stat_forecast", "priority"]


def obj(props: dict, required=None, extra=True) -> dict:
    return {"type": "object", "properties": props, "required": list(required if required is not None else props),
            "additionalProperties": extra}


def arr(items, min_items=0) -> dict:
    return {"type": "array", "items": items, "minItems": min_items}


METRIC_ROW = obj({"stage": STR, "model": MODEL_ENUM, "variant": STR, "slice_type": STR, "slice_value": STR,
                  "wape": NUMN, "bias": NUMN, "mae": NUMN, "rmse": NUMN, "n": INT})
TRUST_COMPONENTS = obj({k: NUM for k in ["accuracy", "interval", "stability", "agreement", "completeness"]})

SCHEMAS = {
    "meta.json": obj({
        "generated_at": STR, "citation": STR, "caveats": arr(STR, 1),
        "data_window": obj({"start": DATE, "last_actual": DATE, "forward_start": DATE, "forward_end": DATE}),
        "split_calendar": obj({"selection": DATES, "calibration": DATES, "holdout": DATES}),
        "scope": obj({"n_stores": INT, "n_families": INT, "n_series": INT, "families": arr(FAM_ENUM, 8),
                      "stores": arr(obj({"store_nbr": INT, "city": STR, "state": STR, "type": STR, "cluster": INT}))}),
        "hero": arr(obj({"label": STR, "value": STR, "detail": STR}), 3),
        "headline": STR,
    }),
    "overview.json": obj({
        "kpis": obj({
            "forecast_14d_total": obj({"value": NUM, "start": DATE, "end": DATE}),
            "holdout_wape": obj({"value": NUM, "model": STR}),
            "holdout_bias": obj({"value": NUM, "model": STR}),
            "review_exceptions": obj({"value": INT, "of": INT}),
            "interval_coverage": obj({"value": NUM, "target": NUM}),
        }),
        "outlook": obj({
            "scale": STR,
            "history": arr(obj({"date": DATE, "mean_index": NUMN})),
            "forecast": arr(obj({"date": DATE, "mean_index": NUM})),
            "chain_band_holdout_coverage": NUM, "band_note": STR,
        }),
        "top_exceptions": arr(obj({"series_id": STR, "store_nbr": INT, "family": FAM_ENUM, "trust_score": NUM,
                                   "band": BAND_ENUM, "reason": STR, "priority_rank": INT})),
        "ml_wins": obj({"families_won": INT, "n_families": INT,
                        "champions": {"type": "object", "additionalProperties": MODEL_ENUM}}),
        "findings": arr(STR, 1),
    }),
    "series_index.json": arr(obj({"series_id": STR, "slug": STR, "store_nbr": INT, "family": FAM_ENUM,
                                  "city": STR, "type": STR, "cluster": INT}), 1),
    "demand.json": obj({
        "weekly": obj({"week_start": DATES, "family_index": {"type": "object", "additionalProperties": arr(NUMN)},
                       "definition": STR}),
        "dow_profile": {"type": "object", "additionalProperties": arr(NUM, 7)},
        "volatility": arr(obj({"store_nbr": INT, "family": FAM_ENUM, "cv": NUM, "zero_share": NUM})),
        "promo_bins": {"type": "object", "additionalProperties": arr(obj({"bin": STR, "n_days": INT,
                                                                         "mean_index": NUM, "n_stores": INT}),
                                                                     )},
        "promo_frequency": arr(obj({"store_nbr": INT, "family": FAM_ENUM, "promo_day_share": NUM})),
        "holiday_effect": arr(obj({"family": FAM_ENUM, "change_pct": NUM, "n_holiday_days": INT})),
        "notes": arr(STR),
    }),
    "arena.json": obj({
        "metrics": arr(METRIC_ROW, 1), "mae_unit": STR,
        "champions": {"type": "object", "additionalProperties": obj({"model": MODEL_ENUM, "selection_wape": NUM,
                                                                     "selection_bias": NUM})},
        "variant_comparison": arr(obj({"stage": STR, "family": STR, "wape_A": NUMN, "wape_B": NUMN})),
        "notes": arr(STR),
    }),
    "intervals.json": obj({
        "method": STR, "level": NUM,
        "quantiles": arr(obj({"family": FAM_ENUM, "bucket": STR, "q_lo": NUM, "q_hi": NUM})),
        "calibration_coverage": NUM,
        "holdout": obj({"coverage": NUM, "mean_width": NUM, "mean_width_pct": NUM,
                        "by_family": arr(obj({"family": FAM_ENUM, "coverage": NUM, "mean_width": NUM,
                                              "mean_width_pct": NUM})),
                        "by_horizon": arr(obj({"horizon": INT, "coverage": NUM, "mean_width": NUM,
                                               "mean_width_pct": NUM}))}),
        "width_unit": STR,
    }),
    "trust.json": obj({
        "weights": TRUST_COMPONENTS, "bands": arr(obj({"min": NUM, "band": BAND_ENUM}), 3),
        "latest_origin": DATE, "scale": STR,
        "latest": arr(obj({"series_id": STR, "store_nbr": INT, "family": FAM_ENUM, "score": NUM, "band": BAND_ENUM,
                           "components": TRUST_COMPONENTS, "hard_flags": arr(STR), "forecast_14d_index": NUM,
                           "p10_14d_index": NUM, "p90_14d_index": NUM, "priority_rank": INT}), 1),
        "validation": obj({
            "n_windows": INT, "spearman": NUMN, "wape_by_band": arr(obj({"band": BAND_ENUM, "n": INT, "wape": NUMN})),
            "error_ratio_review_vs_accept": NUMN, "review_share_of_abs_error": NUMN, "review_share_of_windows": NUMN,
            "volatility_only_spearman": NUMN, "beats_volatility_baseline": BOOL, "verdict": STR}),
        "calibration_choice": obj({"candidates": arr(obj({"name": STR, "spearman": NUMN})), "chosen": STR}),
    }),
    "scenario.json": obj({
        "assumption_label": STR, "baseline_unit": STR,
        "families": arr(obj({"family": FAM_ENUM, "baseline_14d": NUM, "baseline_source": STR}), 1),
        "events": arr(obj({"id": STR, "label": STR, "low": NUM, "base": NUM, "high": NUM}), 1),
        "ecuador_analogue": arr(obj({"family": FAM_ENUM, "change_pct": NUMN, "n_holidays": INT, "window": STR})),
    }),
    "sop.json": obj({
        "origin": DATE, "horizon_end": DATE, "scale": STR,
        "rows": arr(obj({"series_id": STR, "store_nbr": INT, "family": FAM_ENUM, "stat_index": NUM, "p10_index": NUM,
                         "p90_index": NUM, "trust_score": NUM, "trust_band": BAND_ENUM}), 1),
        "reason_codes": arr(STR, 5),
        "summary": obj({"top_risks": arr(STR), "persistent_bias": arr(STR), "review_queue": arr(STR),
                        "promo_opportunities": arr(STR), "discussion_points": arr(STR, 3)}),
        "fva": obj({"label": STR, "rule": STR, "stage": STR, "champion_wape": NUM, "simulated_wape": NUM,
                    "fva_pp": NUM}),
    }),
}

SERIES_SCHEMA = obj({
    "series_id": STR, "store_nbr": INT, "family": FAM_ENUM, "scale": STR, "promo_scale": STR,
    "history": obj({"start": DATE, "index": arr(NUMN), "week_start": DATES, "weekly_index": arr(NUMN)}),
    "holdout": arr(obj({"origin": DATE, "champion": MODEL_ENUM, "rows": arr(obj({
        "date": DATE, "h": INT, "actual_index": NUMN, "p10_index": NUM, "p90_index": NUM, "forecast_index": NUM},
        extra=False), 14)})),
    "forward": obj({"origin": DATE, "champion": MODEL_ENUM, "rows": arr(obj({
        "date": DATE, "h": INT, "forecast_index": NUM, "p10_index": NUM, "p90_index": NUM, "promo_index": NUMN,
        "models_index": {"type": "object", "additionalProperties": NUM}}), 14)}),
    "whatif": obj({"origin": DATE, "hist_max_promo_index": NUMN, "hist_p95_promo_index": NUMN, "rows": arr(obj({
        "scale_pct": INT, "promo_index": NUMN, "forecast_index": NUM, "change_pct": NUMN, "daily_index": arr(NUM, 14),
        "extrapolation": BOOL}), 1)}),
    "trust": obj({"score": NUM, "band": BAND_ENUM, "components": TRUST_COMPONENTS, "hard_flags": arr(STR)}),
})
