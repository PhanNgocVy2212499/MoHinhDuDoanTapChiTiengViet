# Phân Loại Chủ Đề Tin Tức Tiếng Việt

## 1. Giới thiệu dự án

Dự án này xây dựng và so sánh hai mô hình baseline cho bài toán phân loại chủ đề tin tức tiếng Việt.

- Dữ liệu văn bản đầu vào: cột content_cleaned
- Nhãn mục tiêu: cột label
- Tập dữ liệu: dataset_tapchi.csv

Mục tiêu chính của baseline:

- Thiết lập mốc hiệu năng ban đầu trước khi thử các mô hình phức tạp hơn.
- So sánh giữa một mô hình xác suất đơn giản (Naive Bayes) và một mô hình phân lớp biên mạnh (Linear SVM).
- Đánh giá cân bằng giữa độ chính xác và thời gian huấn luyện.

## 2. Cài đặt môi trường

Yêu cầu:

- Python 3.10 trở lên
- Khuyến nghị dùng môi trường ảo để tách biệt thư viện

### 2.1 Tạo và kích hoạt môi trường ảo (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.2 Cài thư viện cần thiết

```powershell
pip install pandas scikit-learn matplotlib seaborn
```

## 3. Cấu trúc script chính

- train_nb_progress.py: huấn luyện riêng mô hình Multinomial Naive Bayes, in tiến độ theo từng bước, in metrics và lưu confusion matrix.
- train_svm_progress.py: huấn luyện riêng mô hình SVM tuyến tính, in tiến độ theo từng bước, in metrics và lưu confusion matrix.
- RunTrain.py: chạy tuần tự NB rồi SVM, gom kết quả và xuất bảng, biểu đồ so sánh.

## 4. Cách chạy

### 4.1 Chạy từng mô hình riêng lẻ

```powershell
python train_nb_progress.py --csv dataset_tapchi.csv --save-cm confusion_matrix_nb.png
python train_svm_progress.py --csv dataset_tapchi.csv --save-cm confusion_matrix_svm.png
```

### 4.2 Chạy tổng hợp 2 mô hình (khuyến nghị)

```powershell
python RunTrain.py --csv dataset_tapchi.csv --out-prefix baseline_compare
```

Sau khi chạy xong, hệ thống sinh ra:

- baseline_compare_metrics.csv: bảng chỉ số tổng hợp
- baseline_compare_summary.md: tóm tắt kết quả
- baseline_compare_charts.png: biểu đồ so sánh metrics và thời gian

## 5. Quy trình xử lý dữ liệu và huấn luyện

Hai mô hình dùng cùng một pipeline tiền xử lý để đảm bảo so sánh công bằng:

1. Đọc dữ liệu từ dataset_tapchi.csv.
2. Chọn hai cột: content_cleaned và label.
3. Loại bỏ dòng thiếu dữ liệu.
4. Chia train/test theo tỷ lệ 80/20 với random_state=42 và stratify theo nhãn.
5. Chuyển văn bản thành vector TF-IDF.
6. Huấn luyện mô hình.
7. Dự đoán trên tập test.
8. Đánh giá bằng Accuracy, Precision weighted, Recall weighted, F1-score weighted.
9. Vẽ confusion matrix để quan sát lỗi nhầm lẫn giữa các chủ đề.

## 6. Mô hình hoạt động ra sao?

## 6.1 Multinomial Naive Bayes

Naive Bayes dựa trên định lý Bayes với giả định các đặc trưng độc lập có điều kiện theo lớp.
Trong bài toán văn bản, mô hình ước lượng xác suất tài liệu thuộc từng lớp dựa trên phân bố từ/ngữ trong lớp đó.

Về trực giác:

- Nếu một tài liệu chứa nhiều từ xuất hiện mạnh ở lớp Kinh doanh, xác suất vào lớp Kinh doanh sẽ tăng.
- Mô hình chọn lớp có xác suất hậu nghiệm lớn nhất.

Ưu điểm:

- Huấn luyện rất nhanh.
- Tốn ít tài nguyên.
- Dễ triển khai, dễ làm mốc baseline.

Hạn chế:

- Giả định độc lập giữa các từ khá mạnh, không phản ánh tốt ngữ nghĩa phức tạp.
- Dễ giảm hiệu quả khi các lớp chồng lấn về từ vựng.

## 6.2 SVM tuyến tính (Linear SVM)

SVM tìm một siêu phẳng phân tách các lớp sao cho khoảng cách biên là lớn nhất.
Với dữ liệu văn bản có số chiều cao và dạng sparse (TF-IDF), Linear SVM thường hoạt động rất hiệu quả.

Về trực giác:

- Mỗi tài liệu là một điểm trong không gian đặc trưng rất lớn.
- SVM học các trọng số để đẩy điểm của lớp này về một phía, lớp khác về phía còn lại của biên quyết định.

Ưu điểm:

- Khả năng tổng quát hóa tốt trên bài toán phân loại văn bản.
- Thường cho F1 và Accuracy cao hơn Naive Bayes trong thực tế.

Hạn chế:

- Huấn luyện chậm hơn Naive Bayes.
- Cần tài nguyên tính toán lớn hơn.

## 7. Kết quả thực nghiệm baseline

Số liệu dưới đây lấy từ chart/baseline_compare_metrics.csv:

| Mô hình      | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) | Thời gian (giây) |
| ------------ | -------- | -------------------- | ----------------- | ------------------- | ---------------- |
| Naive Bayes  | 0.6963   | 0.7057               | 0.6963            | 0.6525              | 71.77            |
| SVM (Linear) | 0.7995   | 0.7925               | 0.7995            | 0.7945              | 126.42           |

Phân tích chi tiết:

- Accuracy: SVM cao hơn đáng kể, gần mốc 80%, cho thấy mô hình phân lớp tổng thể tốt hơn.
- F1-score weighted: SVM vượt trội so với NB, phản ánh sự cân bằng Precision/Recall tốt hơn trên dữ liệu nhiều lớp.
- Recall weighted: SVM nhận diện đúng mẫu thuộc các lớp tốt hơn trung bình toàn tập.
- Thời gian huấn luyện: NB nhanh hơn nhiều, phù hợp khi cần mô hình nhẹ và thời gian chạy ngắn.

Ý nghĩa thực tế:

- Nếu ưu tiên hiệu năng dự đoán: SVM là lựa chọn baseline mạnh hơn.
- Nếu ưu tiên tốc độ huấn luyện hoặc tài nguyên thấp: NB vẫn hữu ích làm mốc ban đầu.

## 8. Kết luận và hướng mở rộng

Kết luận hiện tại:

- Naive Bayes: nhanh, đơn giản, hiệu năng ở mức trung bình.
- SVM tuyến tính: hiệu năng cao hơn rõ rệt, phù hợp làm baseline chính cho báo cáo.

Hướng mở rộng tiếp theo:

- Tối ưu TF-IDF: thử ngram_range=(1,2), tinh chỉnh min_df, max_df.
- Tối ưu SVM: tinh chỉnh tham số C bằng GridSearchCV.
- Phân tích lỗi sâu hơn bằng confusion matrix theo từng lớp bị nhầm mạnh.
- Thử mô hình Deep Learning khi có đủ tài nguyên (GPU, số epoch lớn hơn, dữ liệu train đầy đủ).
