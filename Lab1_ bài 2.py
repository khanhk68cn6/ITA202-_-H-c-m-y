# Bài 2:Phân loại bài toán Machine Learning

# 1. Dự đoán giá nhà dựa trên diện tích.
# -> Supervised Learning (Hồi quy - Regression): Dự đoán một giá trị liên tục (giá nhà) dựa trên nhãn có sẵn.

# 2. Gom nhóm khách hàng dựa trên hành vi mua sắm.
# -> Unsupervised Learning (Phân cụm - Clustering): Tìm cấu trúc ẩn để nhóm dữ liệu chưa gắn nhãn thành các cụm.

# 3. Phân loại email là Spam hay không.
# -> Supervised Learning (Phân loại - Classification): Phân dữ liệu vào các lớp/danh mục có sẵn (Spam hoặc Không Spam).

# 4. Nhận diện chữ viết tay từ hình ảnh.
# -> Supervised Learning (Phân loại - Classification): Dự đoán nhãn của hình ảnh thuộc về ký tự hay chữ số nào dựa trên dữ liệu mẫu đã gắn nhãn.

# Ví dụ 1
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Chuẩn bị dữ liệu mẫu (X: Diện tích m2, y: Giá nhà tỷ VNĐ)
# X cần là mảng 2 chiều trong scikit-learn
X = np.array([[30], [45], [60], [80], [100], [120]]) 
y = np.array([1.5,  2.1,  3.0,  3.8,  5.0,  6.2])

# 2. Khởi tạo mô hình Hồi quy tuyến tính
model = LinearRegression()

# 3. Huấn luyện mô hình với dữ liệu đã gắn nhãn (Học có giám sát)
model.fit(X, y)

# 4. Dự đoán giá nhà cho các diện tích mới chưa từng thấy
dien_tich_moi = np.array([[50], [75], [110]])
gia_du_doan = model.predict(dien_tich_moi)

# 5. In kết quả ra màn hình
print("--- KẾT QUẢ DỰ ĐOÁN GIÁ NHÀ (SUPERVISED) ---")
for dt, gia in zip(dien_tich_moi.flatten(), gia_du_doan):
    print(f"Diện tích: {dt}m2 -> Giá dự đoán: {gia:.2f} tỷ VNĐ")

# Ví dụ 2:
import numpy as np
from sklearn.cluster import KMeans

# 1. Chuẩn bị dữ liệu khách hàng (Chưa có nhãn/nhóm trước)
# Mỗi dòng đại diện cho 1 khách hàng: [Số lần mua sắm/tháng, Điểm thành viên]
X_khach_hang = np.array([
    [1, 10], [2, 15], [1, 12],  # Nhóm ít mua sắm, điểm thấp
    [8, 85], [9, 90], [7, 78],  # Nhóm mua sắm nhiều, điểm cao
    [2, 8],  [8, 80]
])

# 2. Khởi tạo mô hình K-Means với mục tiêu gom thành 2 nhóm (k=2)
kmeans = KMeans(n_clusters=2, random_state=42)

# 3. Huấn luyện mô hình để tự tìm cấu trúc dữ liệu (Học không giám sát)
kmeans.fit(X_khach_hang)

# 4. Lấy nhãn nhóm (cluster label) mà thuật toán tự phân chia
nhom_gan_cho_khach = kmeans.labels_

# 5. In kết quả phân nhóm dữ liệu ban đầu
print("\n--- KẾT QUẢ PHÂN CỤM KHÁCH HÀNG (UNSUPERVISED) ---")
for i, kh in enumerate(X_khach_hang):
    print(f"Khách hàng {i+1} [Mua sắm: {kh[0]}, Điểm: {kh[1]}] -> Được xếp vào Nhóm: {nhom_gan_cho_khach[i]}")

# 6. Gom nhóm cho một khách hàng mới tinh dựa trên thói quen của họ
khach_hang_moi = np.array([[3, 14]])
nhom_du_doan = kmeans.predict(khach_hang_moi)
print(f"\nKhách hàng mới [Mua sắm: 3, Điểm: 14] -> Hệ thống tự xếp vào Nhóm: {nhom_du_doan[0]}")
