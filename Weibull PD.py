import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =========================================================
# GENERATE SYNTHETIC BANKING DATA
# =========================================================

def generate_synthetic_data():

    n = 2000

    # -----------------------------
    # CUSTOMER MASTER DATABASE
    # -----------------------------
    customers = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "segment": np.random.choice(
            ["Corporate", "SME", "Retail"],
            size=n,
            p=[0.4, 0.35, 0.25]
        ),
        "region": np.random.choice(
            ["North", "South", "West", "East"],
            size=n
        )
    })

    # -----------------------------
    # CORE BANKING DATABASE
    # -----------------------------
    ratings = np.random.choice(
        ["AAA", "A", "BBB", "BB"],
        size=n,
        p=[0.15, 0.35, 0.35, 0.15]
    )

    core_banking = pd.DataFrame({
        "loan_id": range(100000, 100000 + n),
        "customer_id": range(1, n + 1),
        "rating": ratings,
        "ead_core": np.random.gamma(
            shape=5,
            scale=50000,
            size=n
        ).round(2)
    })

    # -----------------------------
    # RISK SYSTEM DATABASE
    # Slightly different EAD values
    # to simulate reconciliation
    # -----------------------------
    risk_system = pd.DataFrame({
        "loan_id": core_banking["loan_id"],
        "ead_risk": (
            core_banking["ead_core"] *
            np.random.normal(1, 0.02, size=n)
        ).round(2)
    })

    # -----------------------------
    # HISTORICAL DEFAULT DATABASE
    # Weibull-driven generation
    # -----------------------------
    weibull_parameters = {
        "AAA": (3.0, 120),
        "A": (2.5, 90),
        "BBB": (2.0, 60),
        "BB": (1.5, 36)
    }

    default_times = []
    default_flags = []

    for rating in ratings:

        shape, scale = weibull_parameters[rating]

        simulated_time = weibull_min.rvs(
            shape,
            scale=scale
        )

        # Simulate observation window
        horizon = 120

        if simulated_time <= horizon:
            default_flags.append(1)
            default_times.append(simulated_time)

        else:
            default_flags.append(0)
            default_times.append(np.nan)

    defaults = pd.DataFrame({
        "loan_id": core_banking["loan_id"],
        "default_flag": default_flags,
        "time_to_default_months": default_times
    })

    # -----------------------------
    # SAVE CSV FILES
    # -----------------------------
    customers.to_csv(
        DATA_DIR / "customer_master.csv",
        index=False
    )

    core_banking.to_csv(
        DATA_DIR / "core_banking.csv",
        index=False
    )

    risk_system.to_csv(
        DATA_DIR / "risk_system.csv",
        index=False
    )

    defaults.to_csv(
        DATA_DIR / "historical_defaults.csv",
        index=False
    )


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    customer_master = pd.read_csv(
        DATA_DIR / "customer_master.csv"
    )

    core_banking = pd.read_csv(
        DATA_DIR / "core_banking.csv"
    )

    risk_system = pd.read_csv(
        DATA_DIR / "risk_system.csv"
    )

    historical_defaults = pd.read_csv(
        DATA_DIR / "historical_defaults.csv"
    )

    return (
        customer_master,
        core_banking,
        risk_system,
        historical_defaults
    )


# =========================================================
# CLEAN DATA
# =========================================================

def clean_data(
    customer_master,
    core_banking,
    risk_system,
    historical_defaults
):

    customer_master = customer_master.drop_duplicates(
        subset=["customer_id"]
    )

    core_banking = core_banking.drop_duplicates(
        subset=["loan_id"]
    )

    risk_system = risk_system.drop_duplicates(
        subset=["loan_id"]
    )

    historical_defaults = historical_defaults.drop_duplicates(
        subset=["loan_id"]
    )

    return (
        customer_master,
        core_banking,
        risk_system,
        historical_defaults
    )


# =========================================================
# RECONCILIATION BETWEEN SYSTEMS
# =========================================================

def reconcile_exposures(
    core_banking,
    risk_system
):

    merged = core_banking.merge(
        risk_system,
        on="loan_id",
        how="outer",
        indicator=True
    )

    merged["ead_difference"] = (
        merged["ead_core"] -
        merged["ead_risk"]
    )

    reconciliation_report = {
        "matched_records":
            int((merged["_merge"] == "both").sum()),

        "core_only_records":
            int((merged["_merge"] == "left_only").sum()),

        "risk_only_records":
            int((merged["_merge"] == "right_only").sum()),

        "material_ead_differences":
            int((merged["ead_difference"].abs() > 1000).sum())
    }

    # Final reconciled exposure
    merged["ead_final"] = (
        merged["ead_risk"]
        .combine_first(merged["ead_core"])
    )

    return merged, reconciliation_report


# =========================================================
# INTEGRATE DATABASES
# =========================================================

def integrate_datasets(
    customer_master,
    reconciled_exposures,
    historical_defaults
):

    df = reconciled_exposures.merge(
        customer_master,
        on="customer_id",
        how="left"
    )

    df = df.merge(
        historical_defaults,
        on="loan_id",
        how="left"
    )

    return df


# =========================================================
# QUALITY CHECKS
# =========================================================

def quality_checks(df):

    report = {
        "rows": len(df),

        "missing_customers":
            int(df["customer_id"].isna().sum()),

        "missing_ratings":
            int(df["rating"].isna().sum()),

        "missing_ead":
            int(df["ead_final"].isna().sum()),

        "duplicate_loans":
            int(df["loan_id"].duplicated().sum())
    }

    return report


# =========================================================
# AGGREGATED PORTFOLIO ANALYSIS
# =========================================================

def aggregate_portfolio(df):

    agg = (
        df.groupby(["rating", "segment"])
        .agg(
            total_ead=("ead_final", "sum"),
            avg_ead=("ead_final", "mean"),
            default_rate=("default_flag", "mean"),
            n_loans=("loan_id", "count")
        )
        .reset_index()
    )

    return agg


# =========================================================
# WEIBULL PD CURVE ESTIMATION
# =========================================================

def estimate_pd_curves(df):

    results = {}

    horizon = np.arange(1, 121)

    plt.figure(figsize=(10, 6))

    for rating in sorted(df["rating"].dropna().unique()):

        subset = df[
            (df["rating"] == rating) &
            (df["default_flag"] == 1)
        ]

        defaults = subset[
            "time_to_default_months"
        ].dropna()

        if len(defaults) < 5:
            continue

        # Weibull fit
        shape, loc, scale = weibull_min.fit(
            defaults,
            floc=0
        )

        results[rating] = {
            "shape": shape,
            "scale": scale
        }

        # Cumulative PD curve
        pd_curve = weibull_min.cdf(
            horizon,
            shape,
            loc=0,
            scale=scale
        )

        plt.plot(
            horizon,
            pd_curve,
            label=rating
        )

    plt.title(
        "Forward Probability of Default Curves"
    )

    plt.xlabel("Months")
    plt.ylabel("Cumulative PD")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "pd_curves.png",
        dpi=150
    )

    plt.close()

    return results


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():

    # Generate sample data only once
    if not (
        DATA_DIR / "customer_master.csv"
    ).exists():

        generate_synthetic_data()

    (
        customer_master,
        core_banking,
        risk_system,
        historical_defaults
    ) = load_data()

    (
        customer_master,
        core_banking,
        risk_system,
        historical_defaults
    ) = clean_data(
        customer_master,
        core_banking,
        risk_system,
        historical_defaults
    )

    reconciled_exp, recon_report = reconcile_exposures(
        core_banking,
        risk_system
    )

    integrated_df = integrate_datasets(
        customer_master,
        reconciled_exp,
        historical_defaults
    )

    qc_report = quality_checks(
        integrated_df
    )

    portfolio_agg = aggregate_portfolio(
        integrated_df
    )

    pd_results = estimate_pd_curves(
        integrated_df
    )

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    integrated_df.to_csv(
        OUTPUT_DIR / "integrated_dataset.csv",
        index=False
    )

    portfolio_agg.to_csv(
        OUTPUT_DIR / "portfolio_aggregation.csv",
        index=False
    )

    # =====================================================
    # PRINT REPORTS
    # =====================================================

    print("\n==============================")
    print("RECONCILIATION REPORT")
    print("==============================")

    for k, v in recon_report.items():
        print(f"{k}: {v}")

    print("\n==============================")
    print("QUALITY CHECKS")
    print("==============================")

    for k, v in qc_report.items():
        print(f"{k}: {v}")

    print("\n==============================")
    print("PORTFOLIO AGGREGATION")
    print("==============================")

    print(
        portfolio_agg.head(10)
        .to_string(index=False)
    )

    print("\n==============================")
    print("WEIBULL PARAMETERS")
    print("==============================")

    for rating, values in pd_results.items():

        print(
            f"{rating} | "
            f"shape={values['shape']:.2f} | "
            f"scale={values['scale']:.2f}"
        )

    print("\nOutputs saved in output/ folder.")


if __name__ == "__main__":
    main()