import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import warnings

warnings.filterwarnings('ignore')

# Thiết lập seed để kết quả có thể tái tạo được
np.random.seed(42)

# 1. Đọc dữ liệu từ file CSV
print("D:\PythonProject2\thuchanh\Data_Number_6.csv")
try:
    # Đọc dữ liệu từ file CSV (nếu file được tải lên)
    df = pd.read_csv('D:\PythonProject2\thuchanh\Data_Number_6.csv')

    # Hiển thị 5 dòng đầu tiên
    print("Đọc dữ liệu thành công!")
    print("\nDữ liệu gốc (5 dòng đầu):")
    print(df.head())

    # Thông tin dữ liệu
    print("\nThông tin dữ liệu:")
    print(df.info())

    print("\nThống kê mô tả:")
    print(df.describe())
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
    print("Vui lòng đảm bảo file được tải lên và có tên 'du_lieu_giao_thong.csv'")
    # Tạo dữ liệu giả lập nếu không đọc được file
    print("\nTạo dữ liệu giả lập để tiếp tục phân tích...")

    # Tạo dữ liệu mẫu dựa trên ví dụ đã cung cấp
    data = {
        'timestamp': pd.date_range(start='5/1/2025', periods=5000, freq='6min'),
        'x_coord': np.random.uniform(0, 5, 5000),
        'y_coord': np.random.uniform(0, 5, 5000),
        'vehicle_type': np.random.choice(['Car', 'Motorcycle', 'Bus'], 5000, p=[0.5, 0.4, 0.1]),
        'speed': np.random.uniform(0.1, 60, 5000),
        'traffic_density': np.random.choice(['Low', 'Medium', 'High'], 5000, p=[0.4, 0.4, 0.2])
    }
    df = pd.DataFrame(data)
    print("Đã tạo dữ liệu giả lập!")
    print(df.head())

# 2. Tiền xử lý dữ liệu
print("\nĐang tiền xử lý dữ liệu...")

# Chuyển đổi tên cột sang tiếng Việt để phù hợp với báo cáo
column_mapping = {
    'timestamp': 'thoi_gian',
    'x_coord': 'toa_do_x',
    'y_coord': 'toa_do_y',
    'vehicle_type': 'loai_phuong_tien',
    'speed': 'toc_do',
    'traffic_density': 'mat_do_giao_thong'
}
df = df.rename(columns=column_mapping)

# Chuyển đổi cột thời gian sang kiểu datetime
if not pd.api.types.is_datetime64_any_dtype(df['thoi_gian']):
    df['thoi_gian'] = pd.to_datetime(df['thoi_gian'])

# Chuyển đổi loại phương tiện sang tiếng Việt
vehicle_mapping = {
    'Car': 'o_to',
    'Motorcycle': 'xe_may',
    'Bus': 'xe_buyt'
}
df['loai_phuong_tien'] = df['loai_phuong_tien'].map(vehicle_mapping)

# Chuyển đổi mật độ giao thông sang tiếng Việt
density_mapping = {
    'Low': 'thap',
    'Medium': 'trung_binh',
    'High': 'cao'
}
df['mat_do_giao_thong'] = df['mat_do_giao_thong'].map(density_mapping)

# Xử lý giá trị null nếu có
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print("\nSố lượng giá trị null trong mỗi cột:")
    print(null_counts)
    df = df.dropna()
    print(f"Đã loại bỏ {null_counts.sum()} hàng có giá trị null")

print("\nDữ liệu sau khi tiền xử lý:")
print(df.head())

# 3. Tạo các đặc trưng mới
print("\nĐang tạo các đặc trưng mới...")


# Thêm đặc trưng "giờ cao điểm"
def is_rush_hour(time):
    hour = time.hour

    # Định nghĩa giờ cao điểm sáng (7:00 - 9:30) và chiều (16:30 - 19:00)
    if (7 <= hour < 10) or (16 <= hour < 19):
        return 1
    else:
        return 0


df['gio_cao_diem'] = df['thoi_gian'].apply(is_rush_hour)

# Tạo đặc trưng "ngày trong tuần" và "giờ trong ngày"
df['ngay_trong_tuan'] = df['thoi_gian'].dt.dayofweek
df['gio_trong_ngay'] = df['thoi_gian'].dt.hour

# Tạo đặc trưng "tỷ lệ xe lớn" (tỷ lệ ô tô và xe buýt so với tổng phương tiện)
# Nhóm theo tọa độ (làm tròn để tạo nhóm không gian)
df['toa_do_x_group'] = np.round(df['toa_do_x'], 1)
df['toa_do_y_group'] = np.round(df['toa_do_y'], 1)
df['hour_group'] = df['gio_trong_ngay']


# Tính tỷ lệ xe lớn cho mỗi nhóm
def calculate_large_vehicle_ratio(group):
    total = len(group)
    large_vehicles = len(group[group['loai_phuong_tien'].isin(['o_to', 'xe_buyt'])])
    return large_vehicles / total if total > 0 else 0


# Tính tỷ lệ xe lớn theo nhóm không gian và thời gian
large_vehicle_ratios = df.groupby(['toa_do_x_group', 'toa_do_y_group', 'hour_group']).apply(
    calculate_large_vehicle_ratio)
large_vehicle_ratios = large_vehicle_ratios.reset_index()
large_vehicle_ratios.columns = ['toa_do_x_group', 'toa_do_y_group', 'hour_group', 'ty_le_xe_lon']

# Kết hợp tỷ lệ xe lớn vào DataFrame chính
df = df.merge(large_vehicle_ratios, on=['toa_do_x_group', 'toa_do_y_group', 'hour_group'], how='left')

print("Các đặc trưng đã tạo:")
print(df[['gio_cao_diem', 'ngay_trong_tuan', 'gio_trong_ngay', 'ty_le_xe_lon']].head())

# 4. Xác định các "điểm tắc nghẽn" bằng phân cụm K-Means
print("\nĐang thực hiện phân cụm để xác định điểm tắc nghẽn...")

# Chuẩn bị dữ liệu cho phân cụm
clustering_data = df[['toa_do_x', 'toa_do_y', 'gio_trong_ngay']].copy()

# Mã hóa mật độ giao thông thành số
density_code_mapping = {'thap': 0, 'trung_binh': 1, 'cao': 2}
clustering_data['mat_do_so'] = df['mat_do_giao_thong'].map(density_code_mapping)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Xác định số lượng cụm tối ưu (elbow method)
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Vẽ đồ thị elbow
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Số lượng cụm')
plt.ylabel('Inertia')
plt.title('Elbow Method cho K-Means')
plt.grid(True)
plt.savefig('elbow_method.png')
plt.close()

# Chọn số lượng cụm = 5 (dựa trên đồ thị elbow và đặc điểm dữ liệu)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_data)

# Phân tích các cụm
cluster_analysis = df.groupby('cluster').agg({
    'toa_do_x': 'mean',
    'toa_do_y': 'mean',
    'toc_do': 'mean',
    'mat_do_giao_thong': lambda x: x.value_counts().index[0],
    'gio_trong_ngay': 'mean',
    'gio_cao_diem': 'mean'
}).reset_index()

cluster_analysis['count'] = df.groupby('cluster').size().values
print("\nPhân tích các cụm:")
print(cluster_analysis)

# Trực quan hóa các cụm
plt.figure(figsize=(12, 8))
for cluster in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['toa_do_x'], cluster_data['toa_do_y'],
                label=f'Cụm {cluster}', alpha=0.7)

plt.scatter(cluster_analysis['toa_do_x'], cluster_analysis['toa_do_y'],
            c='black', marker='X', s=100, label='Tâm cụm')

plt.xlabel('Tọa độ X')
plt.ylabel('Tọa độ Y')
plt.title('Phân cụm các điểm giao thông')
plt.legend()
plt.grid(True)
plt.savefig('clustering_map.png')
plt.close()

# 5. Tính toán chỉ số "mức độ nghiêm trọng của tắc nghẽn"
print("\nĐang tính toán chỉ số mức độ nghiêm trọng của tắc nghẽn...")


# Định nghĩa: dựa trên mật độ cao và tốc độ thấp
def calculate_congestion_severity(row):
    # Mật độ càng cao (2), tốc độ càng thấp, mức độ nghiêm trọng càng cao
    density_score = density_code_mapping.get(row['mat_do_giao_thong'], 0)

    # Tốc độ được chuyển đổi thành thang điểm nghịch (tốc độ càng thấp thì điểm càng cao)
    max_speed = df['toc_do'].max()
    speed_score = 1 - (row['toc_do'] / max_speed)

    # Trọng số: 60% cho mật độ, 40% cho tốc độ
    congestion_severity = (0.6 * density_score / 2) + (0.4 * speed_score)

    # Chuyển đổi thành thang điểm 0-100 để dễ hiểu
    return round(congestion_severity * 100, 2)


df['muc_do_tac_nghen'] = df.apply(calculate_congestion_severity, axis=1)

# Phân tích mức độ tắc nghẽn theo cụm
congestion_by_cluster = df.groupby('cluster').agg({
    'muc_do_tac_nghen': ['mean', 'max', 'min', 'std']
}).reset_index()
print("\nMức độ tắc nghẽn theo cụm:")
print(congestion_by_cluster)

# Vẽ biểu đồ phân phối mức độ tắc nghẽn
plt.figure(figsize=(12, 6))
for cluster in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster]
    sns.kdeplot(cluster_data['muc_do_tac_nghen'], label=f'Cụm {cluster}')

plt.xlabel('Mức độ tắc nghẽn')
plt.ylabel('Mật độ')
plt.title('Phân phối mức độ tắc nghẽn theo cụm')
plt.legend()
plt.grid(True)
plt.savefig('congestion_severity_distribution.png')
plt.close()

# 6. Xây dựng mô hình dự đoán mật độ giao thông
print("\nĐang xây dựng mô hình dự đoán mật độ giao thông...")

# Chuẩn bị dữ liệu cho mô hình
# Chọn các đặc trưng đầu vào và đầu ra
X = df[['toa_do_x', 'toa_do_y', 'gio_trong_ngay', 'ngay_trong_tuan',
        'gio_cao_diem', 'ty_le_xe_lon', 'loai_phuong_tien']]
y = df['mat_do_giao_thong']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng pipeline cho tiền xử lý và mô hình
categorical_features = ['loai_phuong_tien']
numerical_features = ['toa_do_x', 'toa_do_y', 'gio_trong_ngay',
                      'ngay_trong_tuan', 'gio_cao_diem', 'ty_le_xe_lon']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['thap', 'trung_binh', 'cao'],
            yticklabels=['thap', 'trung_binh', 'cao'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.savefig('confusion_matrix.png')
plt.close()

# 7. Tạo bản đồ nhiệt mức độ tắc nghẽn
print("\nĐang tạo bản đồ nhiệt mức độ tắc nghẽn...")
plt.figure(figsize=(12, 10))
# Tạo các bin cho tọa độ x và y
x_bins = pd.cut(df['toa_do_x'], bins=20)
y_bins = pd.cut(df['toa_do_y'], bins=20)

# Tạo pivot table
pivot_table = df.pivot_table(
    values='muc_do_tac_nghen',
    index=y_bins,
    columns=x_bins,
    aggfunc='mean'
)
sns.heatmap(pivot_table, cmap='YlOrRd', annot=False)
plt.title('Bản đồ nhiệt mức độ tắc nghẽn')
plt.xlabel('Tọa độ X')
plt.ylabel('Tọa độ Y')
plt.savefig('congestion_heatmap.png')
plt.close()

# 8. Phân tích mức độ tắc nghẽn theo giờ trong ngày
hourly_congestion = df.groupby('gio_trong_ngay')['muc_do_tac_nghen'].mean().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(hourly_congestion['gio_trong_ngay'], hourly_congestion['muc_do_tac_nghen'],
         marker='o', linestyle='-', linewidth=2)
plt.xlabel('Giờ trong ngày')
plt.ylabel('Mức độ tắc nghẽn trung bình')
plt.title('Mức độ tắc nghẽn theo giờ trong ngày')
plt.xticks(range(0, 24))
plt.grid(True)
plt.savefig('hourly_congestion.png')
plt.close()

# 9. Tạo đặc trưng tần suất tắc nghẽn theo vị trí
print("\nĐang phân tích tần suất tắc nghẽn theo vị trí...")
# Nhóm theo tọa độ đã làm tròn và tính tỷ lệ mật độ cao
congestion_frequency = df.groupby(['toa_do_x_group', 'toa_do_y_group']).apply(
    lambda x: len(x[x['mat_do_giao_thong'] == 'cao']) / len(x)
).reset_index()
congestion_frequency.columns = ['toa_do_x_group', 'toa_do_y_group', 'tan_suat_tac_nghen']

# Hiển thị các điểm có tần suất tắc nghẽn cao
high_congestion_points = congestion_frequency[congestion_frequency['tan_suat_tac_nghen'] > 0.5]
print("\nCác điểm có tần suất tắc nghẽn cao (>50%):")
print(high_congestion_points)

# Vẽ bản đồ tần suất tắc nghẽn
plt.figure(figsize=(12, 10))
plt.scatter(congestion_frequency['toa_do_x_group'],
            congestion_frequency['toa_do_y_group'],
            c=congestion_frequency['tan_suat_tac_nghen'],
            cmap='YlOrRd',
            s=50,
            alpha=0.7)
plt.colorbar(label='Tần suất tắc nghẽn')
plt.xlabel('Tọa độ X')
plt.ylabel('Tọa độ Y')
plt.title('Tần suất tắc nghẽn theo vị trí')
plt.grid(True)
plt.savefig('congestion_frequency_map.png')
plt.close()

# 10. Báo cáo tổng kết
print("\n===== BÁO CÁO PHÂN TÍCH DỮ LIỆU GIAO THÔNG =====")
print(f"Tổng số bản ghi: {len(df)}")
print(f"Phân bố loại phương tiện: {df['loai_phuong_tien'].value_counts(normalize=True).to_dict()}")
print(f"Phân bố mật độ giao thông: {df['mat_do_giao_thong'].value_counts(normalize=True).to_dict()}")
print(f"Tốc độ trung bình: {df['toc_do'].mean():.2f} km/h")
print(f"Mức độ tắc nghẽn trung bình: {df['muc_do_tac_nghen'].mean():.2f}")

print("\nĐiểm tắc nghẽn nghiêm trọng nhất (top 5):")
severe_points = df.sort_values('muc_do_tac_nghen', ascending=False).head(5)
for _, row in severe_points.iterrows():
    print(f"Vị trí: ({row['toa_do_x']:.2f}, {row['toa_do_y']:.2f}), "
          f"Thời gian: {row['thoi_gian']}, "
          f"Mức độ tắc nghẽn: {row['muc_do_tac_nghen']:.2f}, "
          f"Mật độ: {row['mat_do_giao_thong']}, "
          f"Tốc độ: {row['toc_do']:.2f} km/h")

print("\nHiệu suất mô hình dự đoán mật độ giao thông:")
model_accuracy = (y_pred == y_test).mean()
print(f"Độ chính xác: {model_accuracy:.4f}")

# 11. Dự đoán cho một số điểm mới
print("\nDự đoán mật độ giao thông cho một số điểm mới:")
new_points = [
    # Điểm 1: Tọa độ trung tâm vào giờ cao điểm
    {'toa_do_x': df['toa_do_x'].mean(), 'toa_do_y': df['toa_do_y'].mean(),
     'gio_trong_ngay': 8, 'ngay_trong_tuan': 1, 'gio_cao_diem': 1,
     'ty_le_xe_lon': 0.4, 'loai_phuong_tien': 'o_to'},

    # Điểm 2: Điểm có tọa độ X, Y cao vào cuối tuần
    {'toa_do_x': df['toa_do_x'].quantile(0.9), 'toa_do_y': df['toa_do_y'].quantile(0.9),
     'gio_trong_ngay': 15, 'ngay_trong_tuan': 6, 'gio_cao_diem': 0,
     'ty_le_xe_lon': 0.3, 'loai_phuong_tien': 'xe_may'},

    # Điểm 3: Điểm có tọa độ X, Y thấp vào đêm khuya
    {'toa_do_x': df['toa_do_x'].quantile(0.1), 'toa_do_y': df['toa_do_y'].quantile(0.1),
     'gio_trong_ngay': 23, 'ngay_trong_tuan': 2, 'gio_cao_diem': 0,
     'ty_le_xe_lon': 0.2, 'loai_phuong_tien': 'o_to'}
]

new_points_df = pd.DataFrame(new_points)
predictions = model.predict(new_points_df)
for i, point in enumerate(new_points):
    print(f"Điểm {i + 1}: ({point['toa_do_x']:.2f}, {point['toa_do_y']:.2f}), "
          f"Giờ: {point['gio_trong_ngay']}, "
          f"Ngày: {point['ngay_trong_tuan']}, "
          f"Loại phương tiện: {point['loai_phuong_tien']}, "
          f"Dự đoán mật độ: {predictions[i]}")

print("\nPhân tích hoàn tất!")