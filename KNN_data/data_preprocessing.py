import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    #Tải tập dữ liệu
    dataset = pd.read_csv(file_path)
    x = dataset.iloc[:, [2, 3]].values  # Age and Estimated Salary
    y = dataset.iloc[:, 4].values  # Purchased

    #In dữ liệu ra trong tổng quan về dữ liệu có gì
    print("Data info: \n ")
    print(dataset.info())

    #In dữ liệu ra và những thông tin chi tiết trong đó có gì
    print("Data info details: \n ")
    print(dataset.describe())

    #Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

    #Hàm Chuẩn hóa đặc trưng
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test