import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# =========================================================
# SO SANH 3 MO HINH PHAN LOAI CHU DE TIN TUC TIENG VIET
# 1) Multinomial Naive Bayes
# 2) SVM tuyen tinh (LinearSVC)
# 3) Decision Tree
# Du lieu: dataset_tapchi.csv
# Su dung cot dau vao: content_cleaned
# Su dung cot nhan: label
# =========================================================


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Huấn luyện, dự đoán, đo thời gian và trả về các chỉ số của một mô hình."""
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start

    predict_start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - predict_start

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
        "Training Time (s)": train_time,
        "Prediction Time (s)": predict_time,
        "Total Time (s)": train_time + predict_time,
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

    # 3) Trich xuat TF-IDF mot lan de tat ca mo hinh dung chung.
    # Cach nay dam bao so sanh cong bang vi cung bo dac trung dau vao.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=2,
        lowercase=False,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4) Khoi tao 3 mo hinh can so sanh.
    nb_model = MultinomialNB()
    svm_model = LinearSVC(random_state=42)
    dt_model = DecisionTreeClassifier(
        max_depth=50,
        min_samples_split=20,
        random_state=42,
    )

    # Danh sach nhan de hien thi confusion matrix theo thu tu co dinh.
    labels_order = sorted(y.unique())

    # Danh gia tung mo hinh tren cung X_train_tfidf, y_train.
    nb_metrics, y_pred_nb = evaluate_model(
        "Multinomial Naive Bayes",
        nb_model,
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
    )
    svm_metrics, y_pred_svm = evaluate_model(
        "SVM (LinearSVC)",
        svm_model,
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
    )
    dt_metrics, y_pred_dt = evaluate_model(
        "Decision Tree",
        dt_model,
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
    )

    # 5) In metric chi tiet cua tung mo hinh.
    for metrics in [nb_metrics, svm_metrics, dt_metrics]:
        print("\n" + "=" * 85)
        print(f"KET QUA MO HINH: {metrics['Model']}")
        print("=" * 85)
        print(f"Accuracy             : {metrics['Accuracy']:.4f}")
        print(f"Precision (weighted) : {metrics['Precision (weighted)']:.4f}")
        print(f"Recall (weighted)    : {metrics['Recall (weighted)']:.4f}")
        print(f"F1-score (weighted)  : {metrics['F1-score (weighted)']:.4f}")
        print(f"Training Time (s)    : {metrics['Training Time (s)']:.4f}")
        print(f"Prediction Time (s)  : {metrics['Prediction Time (s)']:.4f}")

    # 6) Bang so sanh tong hop 3 mo hinh (do chinh xac + thoi gian thuc thi).
    results_df = pd.DataFrame([nb_metrics, svm_metrics, dt_metrics])
    numeric_cols = [
        "Accuracy",
        "Precision (weighted)",
        "Recall (weighted)",
        "F1-score (weighted)",
        "Training Time (s)",
        "Prediction Time (s)",
        "Total Time (s)",
    ]
    results_df[numeric_cols] = results_df[numeric_cols].round(4)

    print("=" * 85)
    print("BANG SO SANH 3 MO HINH TREN TAP TEST")
    print("=" * 85)
    print(results_df.to_string(index=False))

    # 7) Ve confusion matrix cho 3 mo hinh de so sanh truc quan.
    cm_nb = confusion_matrix(y_test, y_pred_nb, labels=labels_order)
    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=labels_order)
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=labels_order)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(27, 7))

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

    sns.heatmap(
        cm_dt,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=axes[2],
    )
    axes[2].set_title("Ma tran nham lan - Decision Tree", fontsize=12)
    axes[2].set_xlabel("Nhan du doan")
    axes[2].set_ylabel("Nhan thuc te")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].tick_params(axis="y", rotation=0)

    fig.suptitle("So sanh confusion matrix giua 3 mo hinh", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
