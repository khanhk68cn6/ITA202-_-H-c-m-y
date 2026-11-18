#Bài 3: Trực quan hóa kết quả
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Chuẩn bị dữ liệu (lấy 1 đặc trưng BMI từ tập diabetes)
diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

# Chia dữ liệu Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Huấn luyện mô hình và dự đoán trên tập Test
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Trực quan hóa kết quả bằng Matplotlib
plt.figure(figsize=(10, 6))  # Thiết lập kích thước biểu đồ

# Vẽ biểu đồ tán xạ (scatter plot) cho dữ liệu thực tế trên tập Test
plt.scatter(
    X_test,
    y_test,
    color="royalblue",
    alpha=0.7,
    edgecolors="k",
    s=60,
    label="Dữ liệu thực tế (Actual)",
)

# Vẽ đường thẳng hồi quy (đường dự đoán) lên cùng biểu đồ
plt.plot(
    X_test,
    y_pred,
    color="crimson",
    linewidth=2.5,
    label="Đường hồi quy dự đoán (Predicted)",
)

# Trang trí biểu đồ (tiêu đề, nhãn trục, lưới, chú thích)
plt.title("So sánh Dữ liệu thực tế và Đường hồi quy tuyến tính", fontsize=14, fontweight="bold")
plt.xlabel("Chỉ số BMI (đã chuẩn hóa)", fontsize=12)
plt.ylabel("Chỉ số tiến triển bệnh tiểu đường", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)  # Thêm lưới nền cho dễ quan sát
plt.legend(fontsize=11, loc="upper left")   # Hiển thị hộp chú thích

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
