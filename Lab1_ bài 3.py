# Bài  3: Chạy thử mô hình đơn giản với Scikit-Learn
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# 1. Load dữ liệu Iris
# Dữ liệu gồm 4 tính năng (chiều dài/rộng đài hoa và cánh hoa) 
# Nhãn gốc (target) là các số 0, 1, 2 đại diện cho 3 loài hoa khác nhau
iris = load_iris()
X = iris.data
y = iris.target

# 2. Khởi tạo mô hình
# Theo yêu cầu, chúng ta sử dụng Linear Regression (Hồi quy tuyến tính)

model = LinearRegression()

# 3. Fit mô hình (Huấn luyện)
# Mô hình học mối quan hệ giữa các đặc trưng hoa (X) và nhãn loài hoa (y)
model.fit(X, y)
print("Đã huấn luyện mô hình thành công!\n")

# 4. Predict (Dự đoán)
# Thử lấy luôn 3 mẫu dữ liệu đầu tiên trong tập Iris để mô hình chạy dự đoán
mau_thu_nghiem = X[:3]
ket_qua_du_doan = model.predict(mau_thu_nghiem)

# In kết quả so sánh giữa Nhãn thực tế và Giá trị dự đoán của Linear Regression
print("--- KẾT QUẢ CHẠY THỬ MÔ HÌNH ---")
for i, (thuc_te, du_doan) in enumerate(zip(y[:3], ket_qua_du_doan)):
    print(f"Mẫu hoa thứ {i+1}:")
    print(f"  - Nhãn thực tế (Gốc): {thuc_te}")
    print(f"  - Giá trị mô hình đoán: {du_doan:.4f} (Do Linear Regression xuất ra số liên tục)")

# Lưu ý: Vì đây là mô hình hồi quy nên kết quả dự đoán sẽ trả về số thực liên tục, 
# chứ không phải số nguyên (phân loại rõ ràng 0, 1, 2) như mô hình phân loại (Classification).