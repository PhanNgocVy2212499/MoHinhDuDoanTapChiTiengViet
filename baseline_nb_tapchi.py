import argparse
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline phân loại chủ đề tin tức tiếng Việt bằng Multinomial Naive Bayes"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="dataset_tapchi.csv",
        help="Đường dẫn file CSV dữ liệu (mặc định: dataset_tapchi.csv)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text_clean",
        help="Tên cột chứa văn bản đã tiền xử lý",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Tên cột nhãn/chủ đề",
    )
    parser.add_argument(
        "--save-cm",
        type=str,
        default="",
        help="Nếu truyền đường dẫn, ma trận nhầm lẫn sẽ được lưu ra file ảnh",
    )
    return parser.parse_args()


def load_data(csv_path: Path, text_col: str, label_col: str) -> tuple[pd.DataFrame, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")

    df = pd.read_csv(csv_path)

    # Tự động dò cột văn bản nếu người dùng giữ mặc định text_clean
    # nhưng dataset dùng tên khác như content_cleaned.
    if text_col not in df.columns and text_col == "text_clean":
        fallback_text_cols = ["content_cleaned", "content", "title"]
        for candidate in fallback_text_cols:
            if candidate in df.columns:
                print(
                    f"Không tìm thấy cột 'text_clean'. Tự động dùng cột văn bản: '{candidate}'."
                )
                text_col = candidate
                break

    required_cols = {text_col, label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Thiếu cột bắt buộc trong dữ liệu: {sorted(missing)}. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    # Loại bỏ dòng thiếu dữ liệu ở 2 cột chính để mô hình học ổn định.
    data = df[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str)
    data[label_col] = data[label_col].astype(str)
    return data, text_col


def train_and_evaluate(data: pd.DataFrame, text_col: str, label_col: str) -> tuple[Pipeline, pd.Series, list[str]]:
    X = data[text_col]
    y = data[label_col]

    # 1) Chia train/test 80/20, cố định random_state để tái lập kết quả.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 2) + 3) TF-IDF + MultinomialNB trong Pipeline để tránh rò rỉ dữ liệu.
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 1),
                    min_df=2,
                    lowercase=False,
                ),
            ),
            ("nb", MultinomialNB()),
        ]
    )

    model.fit(X_train, y_train)

    # 4) Dự đoán và in các chỉ số tổng thể.
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ")
    print("=" * 70)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # 5) Báo cáo chi tiết theo từng lớp/chủ đề.
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    labels_order = sorted(y.unique())
    return model, y_test, y_pred.tolist(), labels_order


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: list[str],
    labels_order: list[str],
    save_path: str = "",
) -> None:
    # 6) Vẽ ma trận nhầm lẫn với tên nhãn rõ ràng ở cả 2 trục.
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_order,
        yticklabels=labels_order,
    )
    plt.title("Ma trận nhầm lẫn - Baseline Multinomial Naive Bayes", fontsize=14)
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nĐã lưu ảnh ma trận nhầm lẫn tại: {save_path}")

    plt.show()


def main() -> None:
    args = parse_args()

    data, actual_text_col = load_data(Path(args.csv), args.text_col, args.label_col)
    model, y_test, y_pred, labels_order = train_and_evaluate(
        data,
        actual_text_col,
        args.label_col,
    )

    # model được giữ để bạn có thể mở rộng lưu model bằng joblib ở bước tiếp theo.
    _ = model
    plot_confusion_matrix(y_test, y_pred, labels_order, save_path=args.save_cm)


if __name__ == "__main__":
    main()
