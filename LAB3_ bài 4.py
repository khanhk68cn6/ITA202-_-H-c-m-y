#Bài 4: Quy trình ML hoàn chỉnh - Phần 3: Thử nghiệm mô hình
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================================================
# BƯỚC 1: LOAD DỮ LIỆU
# ==============================================================================
housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"Kích thước dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")
print(f"Tên các đặc trưng: {housing.feature_names}\n")

# Chia dữ liệu thành tập Train (80%) và tập Test (20%)
# Lưu ý: Phải chia trước khi Scaling để tránh rò rỉ dữ liệu (Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================================================================
# BƯỚC 2: PREPROCESS (CHUẨN HÓA DỮ LIỆU - SCALING)
# ==============================================================================
scaler = StandardScaler()

# Học tham số (mean, std) trên tập Train và biến đổi tập Train
X_train_scaled = scaler.fit_transform(X_train)

# Chỉ dùng tham số đã học từ tập Train để biến đổi tập Test
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# BƯỚC 3: TRAIN (HUẤN LUYỆN MÔ HÌNH)
# ==============================================================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Dự đoán trên tập Test
y_pred = model.predict(X_test_scaled)

# ==============================================================================
# BƯỚC 4: EVALUATE (ĐÁNH GIÁ MÔ HÌNH)
# ==============================================================================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Căn bậc hai của MSE giúp đưa về cùng đơn vị với y (100.000 USD)
r2 = r2_score(y_test, y_pred)

print("--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ---")
print(f"MSE (Mean Squared Error) : {mse:.4f}")
print(f"RMSE (Root MSE)          : {rmse:.4f} (tương đương ~${rmse*100000:,.0f})")
print(f"R-squared (R2 Score)     : {r2:.4f}")

# ==============================================================================
# BƯỚC 5: VISUALIZE (TRỰC QUAN HÓA MỘT PHẦN KẾT QUẢ)
# ==============================================================================
# Vì dữ liệu có 8 chiều (không thể vẽ đường thẳng 2D thông thường), ta sẽ vẽ:
# 1. Biểu đồ Actual vs Predicted (Giá thực tế vs Giá dự đoán) cho 100 mẫu đầu tiên
# 2. Biểu đồ đường so sánh trực tiếp trên 50 mẫu đầu tiên của tập Test

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Biểu đồ 1: Scatter plot Thực tế vs Dự đoán (lấy 300 điểm đầu cho thoáng)
sample_size = 300
axes[0].scatter(
    y_test[:sample_size],
    y_pred[:sample_size],
    alpha=0.6,
    color="teal",
    edgecolors="k",
)
# Vẽ đường chéo lý tưởng y = x (nơi Giá dự đoán == Giá thực tế)
axes[0].plot([0, 5], [0, 5], color="red", linestyle="--", linewidth=2, label="Dự đoán hoàn hảo (y = x)")
axes[0].set_title(f"Giá thực tế vs Dự đoán ({sample_size} mẫu Test)", fontweight="bold")
axes[0].set_xlabel("Giá nhà thực tế ($100,000)")
axes[0].set_ylabel("Giá nhà dự đoán ($100,000)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Biểu đồ 2: So sánh từng mẫu cụ thể (50 mẫu đầu tiên)
n_compare = 50
indices = np.arange(n_compare)
axes[1].plot(indices, y_test[:n_compare], marker="o", label="Thực tế (Actual)", color="navy")
axes[1].plot(indices, y_pred[:n_compare], marker="x", linestyle="--", label="Dự đoán (Predicted)", color="darkorange")
axes[1].set_title(f"Đối chiếu chi tiết {n_compare} căn nhà đầu tiên trong tập Test", fontweight="bold")
axes[1].set_xlabel("Chỉ số mẫu (Sample Index)")
axes[1].set_ylabel("Giá nhà ($100,000)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
