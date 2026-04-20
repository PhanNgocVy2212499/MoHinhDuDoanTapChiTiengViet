# Giải thích Confusion Matrix cho bài toán phân loại chủ đề tin tức

## 1. Confusion Matrix là gì?

Confusion Matrix (ma trận nhầm lẫn) là bảng dùng để so sánh:

- Nhãn thực tế (ground truth)
- Nhãn mô hình dự đoán

Trong biểu đồ bạn đang vẽ:

- Trục dọc (y): **Nhãn thực tế**
- Trục ngang (x): **Nhãn dự đoán**

Mỗi ô `[i, j]` cho biết số mẫu có nhãn thật là lớp `i` nhưng mô hình dự đoán thành lớp `j`.

---

## 2. Cách đọc nhanh ma trận

- **Ô trên đường chéo chính** (từ trên trái xuống dưới phải): dự đoán đúng.
- **Ô ngoài đường chéo**: dự đoán sai (nhầm giữa các chủ đề).

Ý nghĩa:

- Giá trị trên đường chéo càng lớn, mô hình càng tốt ở lớp tương ứng.
- Nếu một hàng có nhiều giá trị dàn sang cột khác, lớp đó đang bị nhầm nhiều.

---

## 3. Liên hệ với các chỉ số đánh giá

Từ confusion matrix, ta tính được các chỉ số:

- **Accuracy**: tỷ lệ dự đoán đúng trên toàn bộ mẫu.
- **Precision** (theo từng lớp): trong số mẫu dự đoán là lớp đó, bao nhiêu mẫu đúng.
- **Recall** (theo từng lớp): trong số mẫu thực sự thuộc lớp đó, mô hình bắt đúng bao nhiêu.
- **F1-score**: trung bình điều hòa giữa Precision và Recall.

Trong bài của bạn dùng `average='weighted'`, tức là các lớp có nhiều mẫu sẽ có trọng số lớn hơn khi tính trung bình.

---

## 4. Giải thích kết quả Naive Bayes bạn vừa chạy

Từ kết quả terminal:

- Accuracy ~ **0.6963**
- F1-score (weighted) ~ **0.6525**

Quan sát từ báo cáo lớp cho thấy:

- Một số lớp làm tốt (ví dụ: *Thể thao*, *Thế giới*, *Giải trí*).
- Một số lớp bị nhầm mạnh (ví dụ: *Du lịch*, *Văn hóa* có recall thấp).

Điều này thường xuất hiện trong dữ liệu mất cân bằng hoặc khi nội dung giữa các lớp có từ vựng tương đồng.

---

## 5. Cách trình bày trong báo cáo môn học

Bạn có thể viết ngắn gọn như sau:

1. Mô hình đạt Accuracy ở mức khá, nhưng hiệu quả không đồng đều giữa các lớp.
2. Confusion Matrix cho thấy mô hình phân biệt tốt các lớp có đặc trưng từ vựng rõ ràng.
3. Các lớp khó tách biệt có xu hướng bị nhầm sang các lớp gần nghĩa.
4. Vì vậy cần so sánh thêm với SVM để kiểm tra khả năng phân tách biên tốt hơn.

---

## 6. Gợi ý cải thiện (nếu cần mở rộng)

- Dùng `ngram_range=(1,2)` cho TF-IDF để giữ thêm ngữ cảnh cụm từ.
- Cân bằng lớp (class weight hoặc resampling).
- So sánh thêm SVM tuyến tính, Logistic Regression.
- Tinh chỉnh tham số bằng GridSearchCV.
