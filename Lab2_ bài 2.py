#Bài 2: Chuẩn hóa dữ liệu (Scaling)
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Tạo dữ liệu cho từng cột (Tuổi từ 20-60, Thu nhập từ 10tr-100tr)
tuoi = [22, 25, 30, 35, 42, 50, 58]
thu_nhap = [12000000, 18000000, 35000000, 45000000, 60000000, 85000000, 98000000]

# Ghép 2 cột thành mảng 2 chiều (7 dòng, 2 cột)
data = np.column_stack((tuoi, thu_nhap))

# Tắt hiển thị ký hiệu khoa học (e+07) để dễ nhìn số tiền
np.set_printoptions(suppress=True)

print('--- Dữ liệu gốc (Tuổi | Thu nhập) ---')
print(data)

# 2. Chuẩn hóa bằng StandardScaler (Trung bình = 0, Độ lệch chuẩn = 1)
standard_scaler = StandardScaler()
data_standardized = standard_scaler.fit_transform(data)

print('\n--- Kết quả sau khi dùng StandardScaler ---')
print(np.round(data_standardized, 3))

# 3. Chuẩn hóa bằng MinMaxScaler (Đưa giá trị về khoảng 0 đến 1)
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)

print('\n--- Kết quả sau khi dùng MinMaxScaler ---')
print(np.round(data_minmax, 3))

print('\n--- Nhận xét nhanh ---')
print(
    '- Dữ liệu gốc: Thu nhập lớn hơn Tuổi hàng triệu lần -> dễ làm sai lệch mô'
    ' hình.'
)
print(
    '- StandardScaler: Đưa cả 2 cột về cùng thang đo có Trung bình = 0 và Độ lệch'
    ' chuẩn = 1.'
)
print(
    '- MinMaxScaler: Ép toàn bộ giá trị của cả 2 cột nằm gọn trong khoảng từ 0'
    ' đến 1.'
)