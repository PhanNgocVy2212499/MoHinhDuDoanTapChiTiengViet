import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def log_step(step: int, total: int, message: str) -> None:
    print(f"[Buoc {step}/{total}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline SVM tuyen tinh voi hien thi tien do terminal"
    )
    parser.add_argument("--csv", type=str, default="dataset_tapchi.csv")
    parser.add_argument("--save-cm", type=str, default="confusion_matrix_svm.png")
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Hien thi bieu do thay vi chi luu file anh",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_steps = 8
    t0 = time.time()

    # 1) Doc du lieu
    log_step(1, total_steps, "Doc file CSV")
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["content_cleaned", "label"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Thieu cot bat buoc: {missing}. Cac cot hien co: {list(df.columns)}"
        )

    # 2) Chuan bi du lieu
    log_step(2, total_steps, "Lam sach du lieu va tao X, y")
    data = df[required_cols].dropna().copy()
    X = data["content_cleaned"].astype(str)
    y = data["label"].astype(str)
    labels_order = sorted(y.unique())
    print(f"  - So mau su dung: {len(data)}", flush=True)
    print(f"  - So chu de (label): {len(labels_order)}", flush=True)

    # 3) Chia tap train/test
    log_step(3, total_steps, "Chia train/test ti le 80/20")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"  - Train size: {len(X_train)}", flush=True)
    print(f"  - Test size : {len(X_test)}", flush=True)

    # 4) TF-IDF
    log_step(4, total_steps, "Trich xuat dac trung TF-IDF")
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=2, lowercase=False)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"  - So tu vung: {len(vectorizer.vocabulary_)}", flush=True)

    # 5) Train SVM tuyen tinh
    # LinearSVC la trien khai SVM tuyen tinh toi uu cho du lieu van ban lon.
    log_step(5, total_steps, "Huấn luyện mo hinh SVM tuyen tinh")
    model = LinearSVC(random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 6) Du doan va danh gia
    log_step(6, total_steps, "Du doan va tinh metric tren tap test")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 70)
    print("KET QUA BASELINE - SVM TUYEN TINH")
    print("=" * 70)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # 7) Ve confusion matrix
    log_step(7, total_steps, "Ve va luu confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    plt.figure(figsize=(11, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels_order,
        yticklabels=labels_order,
    )
    plt.title("Ma tran nham lan - Baseline SVM tuyen tinh", fontsize=13)
    plt.xlabel("Nhan du doan")
    plt.ylabel("Nhan thuc te")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if args.save_cm:
        plt.savefig(args.save_cm, dpi=300, bbox_inches="tight")
        print(f"  - Da luu anh confusion matrix: {args.save_cm}", flush=True)

    if args.show_plot:
        plt.show()
    else:
        plt.close()

    # 8) Tong ket thoi gian
    elapsed = time.time() - t0
    log_step(8, total_steps, f"Hoan tat. Tong thoi gian: {elapsed:.2f} giay")


if __name__ == "__main__":
    main()
