from data_preprocessing import load_and_preprocess_data
from data_processing import train_and_evaluate ,visualize_results


def main():
    #Tải và chuẩn hóa dữ liệu
    data = load_and_preprocess_data("/Users/danh/Downloads/File down /Dữ liệu KNN/DataNew/Social_Network_Ads.csv")
    X_train, X_test, y_train, y_test = data

    #Khớp mô hình và đánh giá
    cm = train_and_evaluate(X_train, X_test, y_train, y_test)

    #Xuất báo cáo hình ảnh (có thể thêm mã để vẽ đồ thị ở đây nếu cần)
    visualize_results(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()