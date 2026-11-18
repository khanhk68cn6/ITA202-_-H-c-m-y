#Bài 2: Đánh giá mô hình
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Tải tập dữ liệu diabetes và chỉ lấy 1 đặc trưng duy nhất (cột chỉ số 2 - BMI)
diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]  # Giữ dạng mảng 2D (442, 1) cho scikit-learn
y = diabetes.target

# 2. Chia dữ liệu thành tập Train (80%) và tập Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Khởi tạo và huấn luyện mô hình trên tập Train
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán kết quả trên tập Test
y_pred = model.predict(X_test)

# 4. Tính toán các chỉ số đánh giá trên tập Test
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# ==============================================================================
# GIẢI THÍCH Ý NGHĨA CÁC CHỈ SỐ ĐÁNH GIÁ:
#
# 1. MAE (Mean Absolute Error - Sai số tuyệt đối trung bình):
#    - Ý nghĩa: Trung bình độ lệch tuyệt đối giữa giá trị dự đoán và giá trị thực tế.
#    - Đặc điểm: Giữ nguyên đơn vị đo của biến mục tiêu (y), dễ diễn giải trực quan
#      và ít bị ảnh hưởng bởi các điểm ngoại lai (outliers). Giá trị càng gần 0 càng tốt.
#
# 2. MSE (Mean Squared Error - Sai số bình phương trung bình):
#    - Ý nghĩa: Trung bình của bình phương các sai số giữa dự đoán và thực tế.
#    - Đặc điểm: Do bình phương sai số nên chỉ số này "phạt nặng" các sai số lớn
#      (nhạy cảm với outliers). Đơn vị bị bình phương nên khó đọc trực tiếp hơn MAE,
#      thường dùng để tối ưu thuật toán. Giá trị càng nhỏ càng tốt.
#
# 3. R-squared (R2 Score - Hệ số xác định):
#    - Ý nghĩa: Cho biết tỷ lệ phần trăm (%) phương sai (sự biến thiên) của biến mục
#      tiêu y được giải thích bởi mô hình hồi quy (từ đặc trưng X).
#    - Đặc điểm: Giá trị thường nằm trong khoảng [0, 1] (có thể âm nếu mô hình quá tệ).
#      R2 càng gần 1 chứng tỏ mô hình khớp với dữ liệu càng tốt. Ở bài này dùng 1 đặc
#      trưng BMI nên R2 khoảng ~0.23 (tức giải thích được khoảng 23% sự biến thiên).
# ==============================================================================

# (Tùy chọn) Trực quan hóa đường hồi quy trên tập Test
plt.scatter(X_test, y_test, color="black", label="Dữ liệu thực tế (Test)")
plt.plot(X_test, y_pred, color="blue", linewidth=2, label="Đường hồi quy dự đoán")
plt.xlabel("Chỉ số BMI (đã chuẩn hóa)")
plt.ylabel("Chỉ số tiến triển bệnh tiểu đường")
plt.title("Hồi quy tuyến tính trên tập dữ liệu Diabetes (1 đặc trưng)")
plt.legend()
plt.show()
