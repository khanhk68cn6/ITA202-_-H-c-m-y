#Bài 1: Xây dựng mô hình Hồi quy tuyến tính
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Tạo dữ liệu giả lập (y = 2X + 1 + nhiễu)
np.random.seed(42)  # Cố định seed để kết quả tái lập được
X = 2 * np.random.rand(100, 1)  # 100 điểm dữ liệu ngẫu nhiên trong khoảng [0, 2)
noise = np.random.randn(100, 1) * 0.5  # Nhiễu chuẩn (Gaussian noise)
y = 2 * X + 1 + noise

# 2. Khởi tạo và huấn luyện mô hình Hồi quy tuyến tính
model = LinearRegression()
model.fit(X, y)

# 3. Lấy và in ra các hệ số của mô hình
coefficient = model.coef_[0][0]
intercept = model.intercept_[0]

print(f"Hệ số góc (Coefficient / w): {coefficient:.4f}")
print(f"Hệ số chặn (Intercept / b): {intercept:.4f}")
print(f"Phương trình hồi quy tìm được: y = {coefficient:.4f}X + {intercept:.4f}")
