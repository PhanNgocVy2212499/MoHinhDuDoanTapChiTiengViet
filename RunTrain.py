import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chay lan luot train_nb_progress.py va train_svm_progress.py, "
            "sau do tao bang/so do so sanh ket qua"
        )
    )
    parser.add_argument("--csv", type=str, default="dataset_tapchi.csv")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="comparison",
        help="Tien to ten file output tong hop",
    )
    return parser.parse_args()


def run_script(
    script_name: str,
    csv_path: str,
    cm_path: str,
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        script_name,
        "--csv",
        csv_path,
        "--save-cm",
        cm_path,
    ]
    print("\n" + "=" * 90, flush=True)
    print(f"Dang chay: {script_name}", flush=True)
    print("Lenh:", " ".join(cmd), flush=True)
    print("=" * 90, flush=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    all_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        all_lines.append(line)
        # Stream to terminal so user sees full training progress in real time.
        print(line, flush=True)

    return_code = process.wait()
    output_text = "\n".join(all_lines)
    return return_code, output_text


def extract_metrics(output_text: str) -> dict[str, float]:
    patterns = {
        "Accuracy": r"Accuracy\s*:\s*([0-9]*\.?[0-9]+)",
        "Precision (weighted)": r"Precision(?:\s*\(weighted\))?\s*:\s*([0-9]*\.?[0-9]+)",
        "Recall (weighted)": r"Recall(?:\s*\(weighted\))?\s*:\s*([0-9]*\.?[0-9]+)",
        "F1-score (weighted)": r"F1-score(?:\s*\(weighted\))?\s*:\s*([0-9]*\.?[0-9]+)",
        "Thoi gian (giay)": r"Tong thoi gian:\s*([0-9]*\.?[0-9]+)\s*giay",
    }

    result: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if not match:
            raise ValueError(f"Khong tim thay chi so '{key}' trong output.")
        result[key] = float(match.group(1))
    return result


def save_comparison_outputs(df: pd.DataFrame, out_prefix: str) -> None:
    csv_file = f"{out_prefix}_metrics.csv"
    md_file = f"{out_prefix}_summary.md"
    chart_file = f"{out_prefix}_charts.png"

    df_to_save = df.copy()
    metric_cols = [
        "Accuracy",
        "Precision (weighted)",
        "Recall (weighted)",
        "F1-score (weighted)",
        "Thoi gian (giay)",
    ]
    df_to_save[metric_cols] = df_to_save[metric_cols].round(4)
    df_to_save.to_csv(csv_file, index=False, encoding="utf-8-sig")

    # Markdown summary for report.
    best_acc_row = df.loc[df["Accuracy"].idxmax()]
    best_f1_row = df.loc[df["F1-score (weighted)"].idxmax()]
    best_time_row = df.loc[df["Thoi gian (giay)"].idxmin()]

    markdown_table = build_markdown_table(df_to_save)

    md_lines = [
        "# So sanh ket qua huan luyen",
        "",
        "## Bang so sanh",
        "",
        markdown_table,
        "",
        "## Nhan xet nhanh",
        "",
        f"- Mo hinh co Accuracy cao nhat: **{best_acc_row['Model']}** ({best_acc_row['Accuracy']:.4f})",
        f"- Mo hinh co F1-score weighted cao nhat: **{best_f1_row['Model']}** ({best_f1_row['F1-score (weighted)']:.4f})",
        f"- Mo hinh train nhanh hon: **{best_time_row['Model']}** ({best_time_row['Thoi gian (giay)']:.2f} giay)",
        "",
        "## Tep bieu do",
        "",
        f"- {chart_file}",
    ]
    Path(md_file).write_text("\n".join(md_lines), encoding="utf-8")

    # Draw easy-to-read comparison charts.
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metric_plot_cols = [
        "Accuracy",
        "Precision (weighted)",
        "Recall (weighted)",
        "F1-score (weighted)",
    ]
    long_df = df.melt(
        id_vars=["Model"],
        value_vars=metric_plot_cols,
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(data=long_df, x="Metric", y="Score", hue="Model", ax=axes[0])
    axes[0].set_title("So sanh chi so hieu nang")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_xlabel("Chi so")
    axes[0].set_ylabel("Gia tri")

    sns.barplot(
        data=df,
        x="Model",
        y="Thoi gian (giay)",
        hue="Model",
        legend=False,
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("So sanh thoi gian huan luyen")
    axes[1].set_xlabel("Mo hinh")
    axes[1].set_ylabel("Giay")
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle("Tong hop so sanh Naive Bayes va SVM", fontsize=14)
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nDa xuat cac file tong hop:", flush=True)
    print(f"- {csv_file}", flush=True)
    print(f"- {md_file}", flush=True)
    print(f"- {chart_file}", flush=True)


def build_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines.append(header_line)
    lines.append(separator_line)

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if not Path(args.csv).exists():
        raise FileNotFoundError(f"Khong tim thay file du lieu: {args.csv}")

    all_results: list[dict[str, float | str]] = []

    jobs = [
        ("Naive Bayes", "train_nb_progress.py", "confusion_matrix_nb.png"),
        ("SVM (Linear)", "train_svm_progress.py", "confusion_matrix_svm.png"),
    ]

    for model_name, script_name, cm_file in jobs:
        return_code, output_text = run_script(script_name, args.csv, cm_file)
        if return_code != 0:
            raise RuntimeError(
                f"Script {script_name} that bai voi ma loi {return_code}."
            )

        metrics = extract_metrics(output_text)
        row: dict[str, float | str] = {"Model": model_name}
        row.update(metrics)
        all_results.append(row)

    # Summary in terminal.
    result_df = pd.DataFrame(all_results)
    display_df = result_df.copy()
    for col in [
        "Accuracy",
        "Precision (weighted)",
        "Recall (weighted)",
        "F1-score (weighted)",
        "Thoi gian (giay)",
    ]:
        display_df[col] = display_df[col].astype(float).round(4)

    print("\n" + "=" * 90, flush=True)
    print("BANG SO SANH KET QUA CUOI CUNG", flush=True)
    print("=" * 90, flush=True)
    print(display_df.to_string(index=False), flush=True)

    save_comparison_outputs(result_df, args.out_prefix)


if __name__ == "__main__":
    main()
