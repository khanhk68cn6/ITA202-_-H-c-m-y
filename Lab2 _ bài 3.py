#Bài 3: Chia tập dữ liệu Train/Test
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1. Tải tập dữ liệu diabetes từ sklearn.datasets
diabetes = load_diabetes()
X = diabetes.data  # Ma trận đặc trưng (features)
y = diabetes.target  # Biến mục tiêu (target)

# 2. Chia dữ liệu thành tập train (80%) và test (20%)
# random_state=42 giúp cố định cách chia ngẫu nhiên để lần chạy nào cũng ra cùng một kết quả
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. In ra kích thước của tập gốc và hai tập sau khi chia
print('=== KÍCH THƯỚC TẬP DỮ LIỆU GỐC ===')
print(f'Kích thước đặc trưng X: {X.shape} ({X.shape[0]} mẫu, {X.shape[1]} cột)')
print(f'Kích thước nhãn y:      {y.shape}')

print('\n=== KÍCH THƯỚC SAU KHI CHIA (80% TRAIN - 20% TEST) ===')
print(f'Tập Train (80%) -> X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'Tập Test (20%)  -> X_test:  {X_test.shape},  y_test:  {y_test.shape}')

# 4. In đánh giá nhanh và bài học rút ra
print('\n=== ĐÁNH GIÁ & BÀI HỌC RÚT RA ===')
print(
    '1. Tỷ lệ phân chia: 442 mẫu được chia chuẩn xác thành 353 mẫu Train'
    ' (~79.9%) và 89 mẫu Test (~20.1%), giữ nguyên đủ 10 đặc trưng.'
)
print(
    '2. Chống học vẹt (Overfitting): Tách riêng 89 mẫu Test giúp đánh giá'
    ' khách quan mô hình trên dữ liệu chưa từng thấy.'
)
print(
    '3. Nguyên tắc vàng: Luôn chia Train/Test trước khi chuẩn hóa hay xử lý dữ'
    ' liệu để tránh rò rỉ thông tin (Data Leakage).'
)
print(
    '4. Tính tái lập: Tham số random_state=42 cố định cách xáo trộn ngẫu nhiên,'
    ' giúp kết quả đồng nhất ở mọi lần chạy.'
)