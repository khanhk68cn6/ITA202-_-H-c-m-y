#Bài 4: Quy trình ML hoàn chỉnh - Phần 2: Tiền xử lý dữ liệu
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Tải tập dữ liệu digits (1797 ảnh chữ số viết tay 8x8 = 64 pixel, giá trị từ 0-16)
digits = load_digits()
X = digits.data
y = digits.target

# 2. Chia tập dữ liệu thành Train (80%) và Test (20%)
# Lưu ý: Luôn chia dữ liệu TRƯỚC khi chuẩn hóa để tránh rò rỉ dữ liệu (Data Leakage)
# stratify=y giúp giữ tỷ lệ các chữ số (0-9) đồng đều ở cả 2 tập
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Xây dựng Pipeline gồm 2 bước:
# - Bước 1 ('scaler'): Chuẩn hóa giá trị pixel (0-16) về phân phối chuẩn
# - Bước 2 ('model'): Mô hình phân loại Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42)),
])

# 4. Huấn luyện toàn bộ Pipeline trên tập Train
pipeline.fit(X_train, y_train)

# 5. Dự đoán và đánh giá độ chính xác trên tập Test
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('=== KẾT QUẢ QUY TRÌNH ML (PIPELINE) TRÊN TẬP DIGITS ===')
print(f'Kích thước tập Train: {X_train.shape} | Tập Test: {X_test.shape}')
print(
    f'Khoảng giá trị pixel trước chuẩn hóa: [{X_train.min()}, {X_train.max()}]'
)
print(
    f'Độ chính xác (Accuracy) trên tập Test: {accuracy * 100:.2f}%'
    f' ({accuracy:.4f})'
)

# 6. In đánh giá nhanh cho bài làm
print('\n=== ĐÁNH GIÁ NHANH ===')
print(
    '- Chuẩn hóa dữ liệu (StandardScaler) đưa các pixel từ thang đo [0, 16] về'
    ' cùng phân phối chuẩn, giúp mô hình hội tụ nhanh và ổn định hơn.'
)
print(
    '- Pipeline tự động hóa quy trình: tự học tham số chuẩn hóa trên tập Train'
    ' và áp dụng đúng tham số đó lên tập Test, loại bỏ hoàn toàn nguy cơ rò rỉ'
    ' dữ liệu.'
)
print(
    f'- Mô hình Logistic Regression kết hợp chuẩn hóa đạt độ chính xác rất cao'
    f' ({accuracy * 100:.2f}%) khi nhận diện chữ số viết tay.'
)