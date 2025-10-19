import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def load_background_dataframe(repo_root: Path) -> pd.DataFrame:
    input_csv_path = repo_root / "data" / "background-clean.csv"
    return pd.read_csv(input_csv_path)


def get_class_indicator_columns(df: pd.DataFrame) -> List[str]:
    indicator_columns: List[str] = []
    for column_name in df.columns:
        series = pd.to_numeric(df[column_name], errors="coerce")
        if series.isna().all():
            continue
        non_null_values = series.dropna()
        unique_values = set(non_null_values.unique().tolist())
        if unique_values.issubset({0, 1}):
            has_zero = (non_null_values == 0).any()
            has_one = (non_null_values == 1).any()
            if has_zero and has_one:
                indicator_columns.append(column_name)
    return indicator_columns


def coerce_target_to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        lowered = series.astype(str).str.strip().str.lower()
        prof_map = {"beg": 0, "int": 1, "adv": 2}
        allowed = set([v for v in lowered.unique() if v == v])
        if allowed.issubset(set(prof_map.keys()) | {"nan"}):
            series = lowered.map(prof_map)
            return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def compute_class_confidence_associations(
    df: pd.DataFrame,
    class_columns: List[str],
    targets: List[str],
    alpha: float = 0.15,
) -> pd.DataFrame:
    records: List[dict] = []

    for target in targets:
        for class_column in class_columns:
            class_series = pd.to_numeric(df[class_column], errors="coerce")
            target_series = coerce_target_to_numeric(df[target])

            valid_mask = class_series.notna() & target_series.notna()
            if valid_mask.sum() < 3:
                continue

            class_values = class_series.loc[valid_mask]
            target_values = target_series.loc[valid_mask]

            if class_values.nunique() < 2 or target_values.nunique() < 2:
                continue

            # No Pearson correlation retained; analysis focuses on group comparison (Welch's t-test)

            took_mask = valid_mask & (class_series == 1)
            not_took_mask = valid_mask & (class_series == 0)

            took_values = target_series.loc[took_mask]
            not_took_values = target_series.loc[not_took_mask]

            mean_confidence_took = took_values.mean()
            mean_confidence_not_took = not_took_values.mean()

            n_took = int(took_values.count())
            n_not_took = int(not_took_values.count())

            # Welch's t-test for difference in means
            try:
                t_statistic, p_value_ttest = stats.ttest_ind(
                    took_values, not_took_values, equal_var=False, nan_policy="omit"
                )
            except Exception:
                t_statistic, p_value_ttest = (np.nan, np.nan)

            # Cohen's d (pooled SD)
            cohen_d = np.nan
            if n_took >= 2 and n_not_took >= 2:
                sd_took = np.nanstd(took_values, ddof=1)
                sd_not_took = np.nanstd(not_took_values, ddof=1)
                denom_df = (n_took + n_not_took - 2)
                if denom_df > 0:
                    pooled_var = ((n_took - 1) * (sd_took ** 2) + (n_not_took - 1) * (sd_not_took ** 2)) / denom_df
                    pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else np.nan
                    if pooled_sd and np.isfinite(pooled_sd) and pooled_sd > 0:
                        cohen_d = (mean_confidence_took - mean_confidence_not_took) / pooled_sd

            records.append(
                {
                    "class_name": class_column,
                    "target": target,
                    "t_statistic": t_statistic,
                    "p_value_ttest": p_value_ttest,
                    "cohen_d": cohen_d,
                    "mean_confidence_took": mean_confidence_took,
                    "mean_confidence_not_took": mean_confidence_not_took,
                    "mean_diff": (mean_confidence_took - mean_confidence_not_took),
                    "n_took": n_took,
                    "n_not_took": n_not_took,
                }
            )

    result_df = pd.DataFrame.from_records(records)
    if result_df.empty:
        return result_df

    # FDR correction within each target family (Benjamini-Hochberg)
    # Use Welch's t-test p-values for multiple testing control.
    result_df["p_fdr_ttest"] = np.nan
    result_df["significant_fdr_ttest"] = False
    for target in targets:
        scope = (result_df["target"] == target) & result_df["p_value_ttest"].notna() & np.isfinite(result_df["p_value_ttest"])
        if scope.any():
            pvals = result_df.loc[scope, "p_value_ttest"].values
            reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
            result_df.loc[scope, "p_fdr_ttest"] = pvals_corrected
            result_df.loc[scope, "significant_fdr_ttest"] = reject

    # Sort for readability
    result_df = result_df.sort_values(by=["target", "mean_diff"], ascending=[True, False]).reset_index(drop=True)
    return result_df


def write_outputs(repo_root: Path, associations_df: pd.DataFrame) -> Path:
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_csv = results_dir / "class_confidence_associations.csv"
    associations_df.to_csv(output_csv, index=False)
    return output_csv


def build_text_summary(associations_df: pd.DataFrame, targets: List[str], top_k: int = 10) -> str:
    lines: List[str] = []
    for target in targets:
        subset = associations_df[associations_df["target"] == target]
        if subset.empty:
            continue

        pos_sig = subset[(subset["pearson_r"] > 0) & (subset["significant_fdr"] == True)]
        top_pos_sig = pos_sig.head(top_k)

        lines.append(f"Top {len(top_pos_sig)} positively associated classes for {target} (FDR q<0.05):")
        for _, row in top_pos_sig.iterrows():
            class_name = row["class_name"]
            r_value = row["pearson_r"]
            p_value = row["p_value"]
            p_fdr = row["p_fdr"]
            mean_diff = row["mean_diff"]
            n_took = int(row["n_took"])
            n_not_took = int(row["n_not_took"])
            mean_took = row["mean_confidence_took"]
            mean_not_took = row["mean_confidence_not_took"]
            lines.append(
                f"  - {class_name}: r={r_value:.3f}, p={p_value:.3g}, q={p_fdr:.3g}, "
                f"mean_diff={mean_diff:.3f} (took={mean_took:.3f}, not_took={mean_not_took:.3f}), "
                f"n_took={n_took}, n_not_took={n_not_took}"
            )

        lines.append("")

    return "\n".join(lines).strip()


def save_html_summary(repo_root: Path, associations_df: pd.DataFrame, targets: List[str], top_k: int = 10, alpha: float = 0.15) -> Path:
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "class_confidence_summary.html"

    def format_float(value, digits: int = 3):
        if pd.isna(value):
            return ""
        return f"{value:.{digits}f}"

    html_parts: List[str] = []
    html_parts.append("<!doctype html>")
    html_parts.append("<html lang=\"en\">")
    html_parts.append("<head>")
    html_parts.append("<meta charset=\"utf-8\">")
    html_parts.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    html_parts.append("<title>Class-Confidence Associations</title>")
    html_parts.append(
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,\"Helvetica Neue\",Arial,sans-serif;padding:24px;max-width:1000px;margin:auto} h1{margin-top:0} table{border-collapse:collapse;width:100%;margin:12px 0} th,td{border:1px solid #ddd;padding:8px} th{background:#f5f5f5;text-align:left} tr:nth-child(even){background:#fafafa} .small{color:#555;font-size:0.9em}</style>"
    )
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("<h1>Class-Confidence Associations</h1>")
    html_parts.append(f"<p class=\"small\">I compared confidence between students who took each class and those who didn't using Welch's t-test (handles unequal variances). To account for multiple comparisons within each confidence target, I applied Benjamini–Hochberg FDR and flagged classes with q≤{alpha:.2f}. Effect sizes are reported with Cohen's d.</p>")
    html_parts.append("<p class=\"small\">Full results CSV: <code>results/class_confidence_associations.csv</code></p>")

    # Methodology
    html_parts.append("<h2>Methodology</h2>")
    html_parts.append("<ul>")
    html_parts.append("<li>I used our intake survey (<code>data/background-clean.csv</code>). For the proficiency columns (<code>*.prof</code>) I converted beg=0, int=1, adv=2 so we can work with numbers. Confidence columns (<code>*.comf</code>) are already numeric.</li>")
    html_parts.append("<li>I treated each course column that’s 0/1 as a 'took this class' indicator.</li>")
    html_parts.append("<li>For each course and each target (programming, math, stats), I compared the average confidence of students who took the class vs those who didn’t using <strong>Welch’s t-test</strong> (good when groups can have different variances and sizes).</li>")
    html_parts.append(f"<li>Because we’re testing lots of courses, I controlled the false discovery rate using <strong>Benjamini–Hochberg</strong> within each target and highlighted anything with <strong>q≤{alpha:.2f}</strong>.</li>")
    html_parts.append("<li>I also report <strong>Cohen’s d</strong> to show the size of the difference (≈0.2 small, ≈0.5 medium, >0.8 large).</li>")
    html_parts.append("</ul>")

    # Quick summary at top
    html_parts.append("<h2>Summary</h2>")
    any_sig = False
    for target in targets:
        sub = associations_df[(associations_df["target"] == target) & (associations_df["mean_diff"] > 0) & (associations_df["significant_fdr_ttest"] == True)]
        if not sub.empty:
            any_sig = True
            top = sub.sort_values("mean_diff", ascending=False).head(min(5, len(sub)))
            html_parts.append(f"<p><strong>{target}</strong>: {len(sub)} significant class(es) at q≤{alpha:.2f}.</p>")
            html_parts.append("<ul>")
            for _, row in top.iterrows():
                html_parts.append(
                    f"<li>{row['class_name']}: mean_diff={format_float(row['mean_diff'])}, q={format_float(row['p_fdr_ttest'])}, d={format_float(row['cohen_d'])}</li>"
                )
            html_parts.append("</ul>")
    if not any_sig:
        html_parts.append(f"<p>No significant positive associations at q≤{alpha:.2f} across all targets.</p>")

    for target in targets:
        subset = associations_df[associations_df["target"] == target]
        html_parts.append(f"<h2>Target: {target}</h2>")
        if subset.empty:
            html_parts.append("<p>No results for this target.</p>")
            continue

        pos_sig = subset[(subset["mean_diff"] > 0) & (subset["significant_fdr_ttest"] == True)]
        pos_sig = pos_sig.sort_values("mean_diff", ascending=False).head(top_k)

        if pos_sig.empty:
            html_parts.append(f"<p>No significant positive associations at q≤{alpha:.2f}.</p>")
            continue

        html_parts.append("<table>")
        html_parts.append(
            "<thead><tr>"
            "<th>class_name</th>"
            "<th>t (Welch)</th>"
            "<th>p (Welch)</th>"
            "<th>q (FDR, Welch)</th>"
            "<th>Cohen d</th>"
            "<th>mean_diff</th>"
            "<th>took_mean</th>"
            "<th>not_took_mean</th>"
            "<th>n_took</th>"
            "<th>n_not_took</th>"
            "</tr></thead>"
        )
        html_parts.append("<tbody>")
        for _, row in pos_sig.iterrows():
            html_parts.append(
                "<tr>"
                f"<td>{row['class_name']}</td>"
                f"<td>{format_float(row['t_statistic'])}</td>"
                f"<td>{format_float(row['p_value_ttest'])}</td>"
                f"<td>{format_float(row['p_fdr_ttest'])}</td>"
                f"<td>{format_float(row['cohen_d'])}</td>"
                f"<td>{format_float(row['mean_diff'])}</td>"
                f"<td>{format_float(row['mean_confidence_took'])}</td>"
                f"<td>{format_float(row['mean_confidence_not_took'])}</td>"
                f"<td>{int(row['n_took'])}</td>"
                f"<td>{int(row['n_not_took'])}</td>"
                "</tr>"
            )
        html_parts.append("</tbody>")
        html_parts.append("</table>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    return summary_path


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze class-confidence associations with FDR.")
    parser.add_argument("--alpha", type=float, default=0.15, help="FDR q-value threshold (default: 0.15)")
    parser.add_argument("--top-k", type=int, default=10, help="Top significant classes to show per target in HTML")
    return parser.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    repo_root = Path(__file__).resolve().parents[1]

    df = load_background_dataframe(repo_root)
    print(df)
    class_columns = get_class_indicator_columns(df)
    print(class_columns)
    targets = [
        "prog.prof",
        "prog.comf",
        "math.prof",
        "math.comf",
        "stat.prof",
        "stat.comf",
    ]

    if not targets:
        print("No target confidence columns found.")
        return 1

    associations_df = compute_class_confidence_associations(df, class_columns, targets, alpha=args.alpha)
    if associations_df.empty:
        print("No associations could be computed (insufficient variability or data).")
        return 0

    output_csv = write_outputs(repo_root, associations_df)

    summary_path = save_html_summary(repo_root, associations_df, targets, top_k=args.top_k, alpha=args.alpha)

    print(f"Saved detailed results to: {output_csv}")
    print(f"Saved HTML summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


