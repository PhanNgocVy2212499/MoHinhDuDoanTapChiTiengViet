import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# =========================================================
# SO SANH 2 MO HINH BASELINE PHAN LOAI CHU DE TIN TUC TIENG VIET
# 1) Multinomial Naive Bayes
# 2) SVM tuyen tinh (kernel='linear')
# Du lieu: dataset_tapchi.csv
# Su dung cot dau vao: content_cleaned
# Su dung cot nhan: label
# =========================================================


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Huấn luyện, dự đoán và trả về các chỉ số đánh giá của một mô hình."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Recall (weighted)": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "F1-score (weighted)": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }
    return metrics, y_pred


def main():
    # 1) Doc du lieu tu CSV va chon dung 2 cot theo yeu cau.
    df = pd.read_csv("dataset_tapchi.csv")
    required_cols = ["content_cleaned", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Thieu cot bat buoc: {missing_cols}. Cac cot hien co: {list(df.columns)}"
        )

    data = df[required_cols].dropna().copy()
    X = data["content_cleaned"].astype(str)
    y = data["label"].astype(str)

    # 2) Chia du lieu train/test ti le 80/20, random_state=42.
    # stratify=y giu phan bo nhan giua train va test nhat quan hon.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3) + 4) Tao 2 pipeline: TF-IDF + model phan loai.
    # Dung pipeline de tranh ro ri du lieu va de tai su dung quy trinh.
    nb_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 1),
                    min_df=2,
                    lowercase=False,
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )

    svm_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 1),
                    min_df=2,
                    lowercase=False,
                ),
            ),
            ("clf", SVC(kernel="linear", random_state=42)),
        ]
    )

    # Danh sach nhan de hien thi confusion matrix theo thu tu co dinh.
    labels_order = sorted(y.unique())

    # Danh gia tung mo hinh tren cung tap test.
    nb_metrics, y_pred_nb = evaluate_model(
        "Multinomial Naive Bayes", nb_pipeline, X_train, X_test, y_train, y_test
    )
    svm_metrics, y_pred_svm = evaluate_model(
        "SVM (Linear Kernel)", svm_pipeline, X_train, X_test, y_train, y_test
    )

    # 5) In bang so sanh cac chi so tong the.
    results_df = pd.DataFrame([nb_metrics, svm_metrics])
    numeric_cols = [
        "Accuracy",
        "Precision (weighted)",
        "Recall (weighted)",
        "F1-score (weighted)",
    ]
    results_df[numeric_cols] = results_df[numeric_cols].round(4)

    print("=" * 85)
    print("BANG SO SANH HIEU SUAT 2 MO HINH TREN TAP TEST")
    print("=" * 85)
    print(results_df.to_string(index=False))

    # 6) + 7) Ve confusion matrix cho 2 mo hinh de so sanh truc quan.
    cm_nb = confusion_matrix(y_test, y_pred_nb, labels=labels_order)
    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=labels_order)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cm_nb,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=axes[0],
    )
    axes[0].set_title("Ma tran nham lan - Multinomial Naive Bayes", fontsize=12)
    axes[0].set_xlabel("Nhan du doan")
    axes[0].set_ylabel("Nhan thuc te")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_svm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=axes[1],
    )
    axes[1].set_title("Ma tran nham lan - SVM kernel tuyen tinh", fontsize=12)
    axes[1].set_xlabel("Nhan du doan")
    axes[1].set_ylabel("Nhan thuc te")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    fig.suptitle("So sanh confusion matrix giua 2 mo hinh baseline", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
