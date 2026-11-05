#Bài 4: Quy trình ML hoàn chỉnh - Phần 1: Thu thập và Kiểm tra dữ liệu

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. Tải tập dữ liệu 'digits'
digits = load_digits()
X = digits.data  # Dữ liệu đặc trưng (mảng phẳng 64 phần tử cho mỗi ảnh)
y = digits.target  # Nhãn tương ứng (các số từ 0 đến 9)

# 2. In ra hình dạng (shape) của dữ liệu và nhãn
print("--- KIỂM TRA HÌNH DẠNG DỮ LIỆU (SHAPE) ---")
print(f"Hình dạng của ma trận dữ liệu (X): {X.shape}")
print(f"Hình dạng của mảng nhãn (y): {y.shape}")
print(f"Ý nghĩa: Tập dữ liệu có {X.shape[0]} mẫu ảnh, mỗi ảnh có {X.shape[1]} pixel phẳng.\n")

# 3. In giá trị số của mẫu dữ liệu đầu tiên (Pixel phẳng)
print("--- GIÁ TRỊ CỦA MẪU ĐẦU TIÊN (DẠNG MẢNG 1D) ---")
print(X[0])
print(f"-> Nhãn thực tế của mẫu này là: {y[0]}\n")

# 4. Hiển thị mẫu dữ liệu đầu tiên dưới dạng hình ảnh trực quan
print("--- ĐANG HIỂN THỊ HÌNH ẢNH MINH HỌA... ---")
# digits.images[0] tự động trả về ma trận 8x8 của ảnh đầu tiên thay vì mảng phẳng 64
plt.figure(figsize=(4, 4))
plt.imshow(digits.images[0], cmap='gray_r')  # Sử dụng bảng màu xám ngược để ảnh rõ hơn
plt.title(f"Label goc: {y[0]}")
plt.axis('off')  # Ẩn các trục tọa độ x, y cho đẹp mắt
plt.show()
