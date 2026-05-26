import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


# =========================================================
# SETUP
# =========================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# =========================================================
# RATING SCALE (GRANULAR CREDIT STRUCTURE)
# =========================================================

RATINGS = [
    "AAA", "AA", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-",
    "B", "CCC"
]


# =========================================================
# WEIBULL PARAMETER MAP (DEFAULT INTENSITY STRUCTURE)
# =========================================================

WEIBULL_PARAM_MAP = {
    "AAA":  (3.2, 140),
    "AA":   (3.0, 130),
    "A+":   (2.8, 115),
    "A":    (2.6, 100),
    "A-":   (2.4, 90),

    "BBB+": (2.2, 80),
    "BBB":  (2.0, 70),
    "BBB-": (1.8, 60),

    "BB+":  (1.6, 50),
    "BB":   (1.5, 40),
    "BB-":  (1.4, 35),

    "B":    (1.3, 28),
    "CCC":  (1.2, 18)
}


# =========================================================
# SCENARIOS
# =========================================================

SCENARIO_ADJUSTMENT = {
    "base": 1.0,
    "adverse": 0.85,
    "severe": 0.70
}


# =========================================================
# SYNTHETIC DATA GENERATION (MULTI-SYSTEM BANK STRUCTURE)
# =========================================================

def generate_data():

    n = 6000

    customers = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "segment": np.random.choice(["Corporate", "SME", "Retail"], n, p=[0.4, 0.35, 0.25]),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "industry": np.random.choice(
            ["Manufacturing", "Energy", "Retail", "Tech", "RealEstate"], n
        )
    })

    core = pd.DataFrame({
        "loan_id": range(100000, 100000 + n),
        "customer_id": range(1, n + 1),
        "rating": np.random.choice(RATINGS, n),
        "ead_core": np.random.gamma(5, 50000, n).round(2),
        "interest_rate": np.random.normal(4.5, 1.2, n).round(2),
        "maturity_months": np.random.randint(12, 180, n)
    })

    risk = pd.DataFrame({
        "loan_id": core["loan_id"],
        "ead_risk": (core["ead_core"] * np.random.normal(1, 0.015, n)).round(2)
    })

    defaults = []

    for r in core["rating"]:

        shape, scale = WEIBULL_PARAM_MAP[r]

        t = weibull_min.rvs(shape, scale=scale)

        defaults.append({
            "time_to_default": t if t <= 120 else np.nan,
            "default_flag": 1 if t <= 120 else 0
        })

    defaults = pd.DataFrame(defaults)
    defaults["loan_id"] = core["loan_id"]

    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    core.to_csv(DATA_DIR / "core.csv", index=False)
    risk.to_csv(DATA_DIR / "risk.csv", index=False)
    defaults.to_csv(DATA_DIR / "defaults.csv", index=False)


# =========================================================
# LOAD
# =========================================================

def load():

    return (
        pd.read_csv(DATA_DIR / "customers.csv"),
        pd.read_csv(DATA_DIR / "core.csv"),
        pd.read_csv(DATA_DIR / "risk.csv"),
        pd.read_csv(DATA_DIR / "defaults.csv")
    )


# =========================================================
# CLEANING
# =========================================================

def clean(cust, core, risk, defl):

    return (
        cust.drop_duplicates("customer_id"),
        core.drop_duplicates("loan_id"),
        risk.drop_duplicates("loan_id"),
        defl.drop_duplicates("loan_id")
    )


# =========================================================
# RECONCILIATION (CROSS-SYSTEM CONTROL)
# =========================================================

def reconcile(core, risk):

    df = core.merge(risk, on="loan_id", how="outer")

    df["ead_diff"] = df["ead_core"] - df["ead_risk"]
    df["ead_final"] = df["ead_risk"].fillna(df["ead_core"])

    report = {
        "matched": int(df["ead_risk"].notna().sum()),
        "missing_risk": int(df["ead_risk"].isna().sum()),
        "material_diff": int((df["ead_diff"].abs() > 1000).sum())
    }

    return df, report


# =========================================================
# INTEGRATION LAYER
# =========================================================

def integrate(customers, exposures, defaults):

    df = exposures.merge(customers, on="customer_id", how="left")
    df = df.merge(defaults, on="loan_id", how="left")

    return df


# =========================================================
# QUALITY CHECKS
# =========================================================

def quality_checks(df):

    return {
        "rows": len(df),
        "missing_rating": int(df["rating"].isna().sum()),
        "missing_ead": int(df["ead_final"].isna().sum()),
        "duplicates": int(df["loan_id"].duplicated().sum()),
        "negative_ead": int((df["ead_final"] < 0).sum())
    }


# =========================================================
# WEIBULL CALIBRATION (BY RATING)
# =========================================================

def calibrate(df):

    params = {}

    for r in RATINGS:

        series = df[
            (df["rating"] == r) &
            (df["default_flag"] == 1)
        ]["time_to_default"].dropna()

        if len(series) < 10:
            continue

        shape, loc, scale = weibull_min.fit(series, floc=0)

        params[r] = {
            "shape": shape,
            "scale": scale,
            "n_obs": len(series)
        }

    return params


# =========================================================
# PD ENGINE (LIFETIME CURVES)
# =========================================================

def pd_engine():

    horizon = np.arange(1, 121)

    curves = {}

    for r in RATINGS:

        shape, scale = WEIBULL_PARAM_MAP[r]

        curves[r] = weibull_min.cdf(
            horizon,
            shape,
            loc=0,
            scale=scale
        )

    return horizon, curves


# =========================================================
# SCENARIO PROJECTION
# =========================================================

def scenario_projection():

    horizon = np.arange(1, 121)

    outputs = {}

    for scen, adj in SCENARIO_ADJUSTMENT.items():

        scen_curves = {}

        for r in RATINGS:

            shape, scale = WEIBULL_PARAM_MAP[r]

            scen_curves[r] = weibull_min.cdf(
                horizon,
                shape,
                loc=0,
                scale=scale * adj
            )

        outputs[scen] = scen_curves

    return horizon, outputs


# =========================================================
# PORTFOLIO AGGREGATION
# =========================================================

def aggregate(df):

    return (
        df.groupby(["rating", "segment"])
        .agg(
            ead=("ead_final", "sum"),
            avg_ead=("ead_final", "mean"),
            default_rate=("default_flag", "mean"),
            n=("loan_id", "count")
        )
        .reset_index()
    )


# =========================================================
# MAIN
# =========================================================

def main():

    logging.info("pipeline start")

    if not (DATA_DIR / "customers.csv").exists():
        generate_data()

    cust, core, risk, defl = load()

    cust, core, risk, defl = clean(cust, core, risk, defl)

    exposures, recon = reconcile(core, risk)

    df = integrate(cust, exposures, defl)

    qc = quality_checks(df)

    calib = calibrate(df)

    horizon, base_curves = pd_engine()

    _, scen_curves = scenario_projection()

    agg = aggregate(df)

    # =====================================================
    # OUTPUTS
    # =====================================================

    pd.DataFrame([recon]).to_csv(OUTPUT_DIR / "recon.csv", index=False)
    pd.DataFrame([qc]).to_csv(OUTPUT_DIR / "qc.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "portfolio.csv", index=False)

    # =====================================================
    # PLOT SCENARIOS
    # =====================================================

    for scen, curves in scen_curves.items():

        plt.figure(figsize=(10, 6))

        for r in RATINGS:
            plt.plot(horizon, curves[r], alpha=0.6)

        plt.title(f"Lifetime PD curves - {scen}")
        plt.xlabel("Months")
        plt.ylabel("Cumulative PD")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"pd_{scen}.png")
        plt.close()

    # =====================================================
    # SUMMARY OUTPUT
    # =====================================================

    logging.info("run completed")

    print(recon)
    print(qc)
    print(agg.head())


if __name__ == "__main__":
    main()
