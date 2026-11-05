#Bài 1: Làm quen với Numpy và Pandas
import numpy as np
import pandas as pd
from sklearn import datasets
# Tạo một mảng Numpy 1 chiều chứa các số từ 1 đến 10.
arr = np.arange(1,11)
print('Mang numpy', arr)
# Tạo một DataFrame từ thư viện Pandas chứa thông tin của 5 sinh viên gồm: Họ tên, Tuổi, Điểm toán.
data = {
    'Họ và tên': ["An", "Khánh", "Cường", "Thanh", "Mạnh"],
    'Tuổi' : [20,21,22,25,19],
    'Điểm toán': [8.0,7.5,9.0,9.5,10]

}
df=pd.DataFrame(data)
print ("\n ---Hiện thị bảng dữ liệu ---")
print(df)

#Hiển thị thông tin tổng quan của DataFrame
print("\n Thông tin tổng quan")
print(df.info())
# tính điểm trung bình môn toán
avg_math = df['Điểm toán'].mean()
print(f"\n Điểm trung bình môn toán: {avg_math}")

