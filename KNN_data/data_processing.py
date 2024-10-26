import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

#Hàm Train dữ liệu
def train_and_evaluate(x_train, x_test, y_train, y_test):
    # Khớp mô hình K-NN
    # Đang chạy với cách tính của Minkowski với p = 2 hoặc p = tự cho
    # p tự cho thì kết quả sẽ khác nhau trên lý thuyết
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(x_train, y_train)

    #Dự đoán kết quả cho tập kiểm tra
    y_pred = classifier.predict(x_test)

    #Tạo ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    #Ký Tự Tương úng của ma trận nhầm lẫn
    labels = ['Not_buy','Buy']

    #Vẽ đồ thị (có thể thêm mã để vẽ đồ thị ở đây nếu cần)

    #1. In ma trận nhầm lẫn
    print("Confusion Matrix: \n")
    for i in range(len(labels)):
        for j in range(len(labels)):
            print(f"{labels[i]} prediction {labels[j]}: {cm[i, j]}")

    visualize_results(x_train, y_train, classifier)

    #Hàm hiễn thị kết quả sau khi đã được train
def visualize_results(x_set, y_set, classifier, filename="/Users/danh/Python/KNN_Basic /KNN_data/visualization.png"):

        # Tạo lưới điểm cho đồ thị
        x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))

        # Dự đoán lớp cho từng điểm trong lưới
        plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                     alpha=0.75, cmap=ListedColormap(("red", "green")))

        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())

        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                        c=ListedColormap(("red", "green"))(i), label=j)

        plt.title("K-NN Algorithm (Training set)")
        plt.xlabel("Age")
        plt.ylabel("Estimated Salary")
        # Lưu hình ảnh vào file
        plt.savefig(filename)  # Lưu hình ảnh với tên file đã chỉ định
        plt.show()# Hiển thị hình ảnh

        if os.path.exists(filename):
            print(f"File hình đã được lưu tại: {filename}")
        else:
            print("Có lỗi xảy ra khi lưu file hình.")