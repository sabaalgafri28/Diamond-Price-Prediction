import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_default_path():
    candidates = [
        "CW1_train.csv",
        "CW1_train (1).csv",
        "CW1_train (2).csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_default_path()
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "Could not find training CSV. Pass a path, e.g. python cw1_eda.py CW1_train.csv"
        )

    df = pd.read_csv(path)

    outcome_col = "outcome" if "outcome" in df.columns else df.columns[0]

    out_dir = "eda_outputs"
    safe_mkdir(out_dir)

    summary_lines = []
    summary_lines.append(f"File: {path}")
    summary_lines.append(f"Rows: {len(df):,}")
    summary_lines.append(f"Columns: {len(df.columns)}")
    summary_lines.append(f"Outcome column: {outcome_col}")

    # Missing values
    missing = df.isna().sum().sort_values(ascending=False)
    missing.to_csv(os.path.join(out_dir, "missing_values.csv"))
    summary_lines.append("Missing values (top 10):")
    summary_lines.extend([f"  {k}: {v}" for k, v in missing.head(10).items()])

    # Duplicate rows
    dup_count = df.duplicated().sum()
    summary_lines.append(f"Duplicate rows: {dup_count}")

    # Outcome summary
    if pd.api.types.is_numeric_dtype(df[outcome_col]):
        summary_lines.append("Outcome summary stats:")
        desc = df[outcome_col].describe()
        summary_lines.extend([f"  {k}: {v}" for k, v in desc.items()])
    else:
        summary_lines.append("Outcome is non-numeric; summary counts:")
        counts = df[outcome_col].value_counts().head(10)
        summary_lines.extend([f"  {k}: {v}" for k, v in counts.items()])

    # Column types
    dtypes = df.dtypes.astype(str)
    dtypes.to_csv(os.path.join(out_dir, "dtypes.csv"))

    # Identify numeric vs categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Outcome distribution plots
    if pd.api.types.is_numeric_dtype(df[outcome_col]):
        plt.figure()
        df[outcome_col].hist(bins=40)
        plt.title(f"Outcome distribution: {outcome_col}")
        plt.xlabel(outcome_col)
        plt.ylabel("Count")
        save_plot(os.path.join(out_dir, "outcome_hist.png"))

        plt.figure()
        plt.boxplot(df[outcome_col].dropna(), vert=False)
        plt.title(f"Outcome boxplot: {outcome_col}")
        plt.xlabel(outcome_col)
        save_plot(os.path.join(out_dir, "outcome_box.png"))

    # Correlations for numeric features
    if outcome_col in num_cols:
        num_df = df[num_cols]
        corrs = num_df.corr(numeric_only=True)[outcome_col].drop(outcome_col)
        corrs = corrs.sort_values(key=lambda s: s.abs(), ascending=False)
        corrs.to_csv(os.path.join(out_dir, "numeric_correlations.csv"))

        top_corrs = corrs.head(15)
        if len(top_corrs) > 0:
            plt.figure(figsize=(8, 5))
            top_corrs.sort_values().plot(kind="barh")
            plt.title("Top numeric correlations with outcome")
            plt.xlabel("Pearson correlation")
            save_plot(os.path.join(out_dir, "top_numeric_correlations.png"))

        # Scatter plots for top numeric features
        top_features = corrs.head(6).index.tolist()
        for col in top_features:
            sample = df[[col, outcome_col]].dropna()
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=123)
            plt.figure()
            plt.scatter(sample[col], sample[outcome_col], s=8, alpha=0.4)
            plt.title(f"Outcome vs {col}")
            plt.xlabel(col)
            plt.ylabel(outcome_col)
            save_plot(os.path.join(out_dir, f"scatter_{col}.png"))

    # Categorical feature analysis
    cat_summary_lines = []
    for col in cat_cols:
        grp = (
            df.groupby(col, dropna=False)[outcome_col]
            .agg(["count", "mean", "median"])
            .sort_values("count", ascending=False)
        )
        grp.to_csv(os.path.join(out_dir, f"cat_{col}_summary.csv"))

        cat_summary_lines.append(f"{col} (top 10 by count):")
        for idx, row in grp.head(10).iterrows():
            cat_summary_lines.append(
                f"  {idx}: count={row['count']}, mean={row['mean']}, median={row['median']}"
            )

        top = grp.head(15)
        if len(top) > 0:
            plt.figure(figsize=(8, 5))
            top["mean"].sort_values().plot(kind="barh")
            plt.title(f"Mean outcome by {col} (top 15 by count)")
            plt.xlabel("Mean outcome")
            plt.ylabel(col)
            save_plot(os.path.join(out_dir, f"mean_outcome_by_{col}.png"))

    # Missingness plot
    if missing.max() > 0:
        top_missing = missing[missing > 0].head(20)
        if len(top_missing) > 0:
            plt.figure(figsize=(8, 5))
            top_missing.sort_values().plot(kind="barh")
            plt.title("Top missing-value columns")
            plt.xlabel("Missing count")
            save_plot(os.path.join(out_dir, "missing_values_top.png"))

    # Write summary
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        f.write("\n\nCategorical summaries:\n")
        f.write("\n".join(cat_summary_lines))

    print(f"EDA complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
