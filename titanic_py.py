import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Đặt các thiết lập để đẹp biểu đồ hơn
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Đọc dữ liệu
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Lưu trữ ID của dữ liệu test để sử dụng khi tạo file kết quả
test_ids = test_data['PassengerId']

# Kết hợp dữ liệu để xử lý
all_data = pd.concat([train_data, test_data], sort=False)

# Xem thông tin dữ liệu
print("Thông tin dữ liệu:")
print(all_data.info())

# Xem các giá trị thiếu
print("\nSố lượng giá trị thiếu trong mỗi cột:")
print(all_data.isna().sum())


# Tạo class để trích xuất thông tin từ tên
class NameFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Trích xuất title từ tên
        X_copy['Title'] = X_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

        # Gộp các title hiếm
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        X_copy['Title'] = X_copy['Title'].replace(rare_titles, 'Rare')
        X_copy['Title'] = X_copy['Title'].replace('Mlle', 'Miss')
        X_copy['Title'] = X_copy['Title'].replace('Ms', 'Miss')
        X_copy['Title'] = X_copy['Title'].replace('Mme', 'Mrs')

        return X_copy


# Class để xử lý cabin
class CabinFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Trích xuất deck từ cabin
        X_copy['Deck'] = X_copy['Cabin'].str.slice(0, 1)
        X_copy['Deck'] = X_copy['Deck'].fillna('U')  # U = Unknown

        return X_copy


# Class để tạo các đặc trưng gia đình
class FamilyFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Tạo FamilySize
        X_copy['FamilySize'] = X_copy['SibSp'] + X_copy['Parch'] + 1

        # Tạo IsAlone
        X_copy['IsAlone'] = (X_copy['FamilySize'] == 1).astype(int)

        # Tạo FarePerPerson
        X_copy['FarePerPerson'] = X_copy['Fare'] / X_copy['FamilySize']

        return X_copy


# Trích xuất đặc trưng từ tên
all_data = NameFeatureExtractor().transform(all_data)

# Kiểm tra phân phối của Title
print("\nPhân phối của Title:")
print(all_data['Title'].value_counts())

# Xử lý đặc trưng từ cabin
all_data = CabinFeatureExtractor().transform(all_data)

# Kiểm tra phân phối của Deck
print("\nPhân phối của Deck:")
print(all_data['Deck'].value_counts())

# Tạo đặc trưng gia đình
all_data = FamilyFeatureExtractor().transform(all_data)

# Kiểm tra phân phối của các đặc trưng gia đình
print("\nThống kê về FamilySize:")
print(all_data['FamilySize'].describe())

print("\nThống kê về IsAlone:")
print(all_data['IsAlone'].value_counts())

print("\nThống kê về FarePerPerson:")
print(all_data['FarePerPerson'].describe())

# 1. Sử dụng mô hình hồi quy tuyến tính để điền giá trị thiếu cho Age
# Đầu tiên, chia dataset thành hai phần: có Age và không có Age
df_with_age = all_data[all_data['Age'].notna()]
df_without_age = all_data[all_data['Age'].isna()]

# Chuẩn bị dữ liệu để xây dựng mô hình dự đoán Age
X_train = pd.get_dummies(df_with_age[['Pclass', 'Sex', 'SibSp', 'Parch', 'Title']], drop_first=True)
y_train = df_with_age['Age']

# Xây dựng mô hình Linear Regression
age_model = LinearRegression()
age_model.fit(X_train, y_train)

# Áp dụng mô hình để dự đoán Age cho các hàng thiếu
X_pred = pd.get_dummies(df_without_age[['Pclass', 'Sex', 'SibSp', 'Parch', 'Title']], drop_first=True)

# Đảm bảo X_pred có cùng cột với X_train
missing_cols = set(X_train.columns) - set(X_pred.columns)
for col in missing_cols:
    X_pred[col] = 0
X_pred = X_pred[X_train.columns]

# Dự đoán Age
age_predictions = age_model.predict(X_pred)

# Điền giá trị Age dự đoán vào dataset
all_data.loc[all_data['Age'].isna(), 'Age'] = age_predictions

# 2. Sử dụng KNN để điền giá trị thiếu cho Embarked
# Chuẩn bị dữ liệu cho KNN
embarked_features = ['Pclass', 'Fare', 'Age']
knn_imputer = KNNImputer(n_neighbors=5)

# Tạo DataFrame tạm thời cho việc điền giá trị thiếu của Embarked
temp_df = all_data.copy()
# One-hot encoding cho Embarked
temp_df = pd.get_dummies(temp_df, columns=['Embarked'], prefix='Emb')

# Chọn các hàng có giá trị Embarked bị thiếu (tất cả các cột Emb_ đều là NaN)
emb_missing_idx = temp_df[temp_df[['Emb_C', 'Emb_Q', 'Emb_S']].isna().all(axis=1)].index

# Nếu có giá trị Embarked bị thiếu
if len(emb_missing_idx) > 0:
    # Sử dụng phương pháp thay thế đơn giản: thay thế bằng giá trị phổ biến nhất
    most_common_embarked = all_data['Embarked'].value_counts().idxmax()
    all_data.loc[all_data['Embarked'].isna(), 'Embarked'] = most_common_embarked
    print(
        f"\nĐã điền {len(emb_missing_idx)} giá trị Embarked bị thiếu bằng giá trị phổ biến nhất: {most_common_embarked}")

# Điền giá trị thiếu cho Fare bằng trung vị theo Pclass
if all_data['Fare'].isna().any():
    # Tính trung vị của Fare theo Pclass
    median_fare = all_data.groupby('Pclass')['Fare'].median()

    # Điền giá trị Fare bị thiếu bằng trung vị theo Pclass tương ứng
    for pclass, median in median_fare.items():
        all_data.loc[(all_data['Fare'].isna()) & (all_data['Pclass'] == pclass), 'Fare'] = median

    print("\nĐã điền giá trị Fare bị thiếu bằng trung vị theo Pclass")

# Tách lại dữ liệu train và test
train_processed = all_data.loc[all_data['Survived'].notna()]
test_processed = all_data.loc[all_data['Survived'].isna()]

# Vẽ biểu đồ boxplot để kiểm tra phân phối của fare_per_person theo pclass và survived
plt.figure(figsize=(12, 6))
sns.boxplot(x='Pclass', y='FarePerPerson', hue='Survived', data=train_processed)
plt.title('Phân phối của FarePerPerson theo Pclass và Survived')
plt.xlabel('Passenger Class')
plt.ylabel('Fare Per Person')
plt.savefig('fare_per_person_boxplot.png')
plt.close()

# Thực hiện t-test để kiểm tra sự khác biệt của FarePerPerson giữa các nhóm survived
survived_fare = train_processed[train_processed['Survived'] == 1]['FarePerPerson'].dropna()
not_survived_fare = train_processed[train_processed['Survived'] == 0]['FarePerPerson'].dropna()

t_stat, p_value = stats.ttest_ind(survived_fare, not_survived_fare, equal_var=False)
print(f"\nKết quả t-test về sự khác biệt của FarePerPerson giữa các nhóm Survived:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(
    f"Kết luận: {'Có sự khác biệt có ý nghĩa thống kê' if p_value < 0.05 else 'Không có sự khác biệt có ý nghĩa thống kê'}")

# Đặc trưng sẽ sử dụng trong mô hình
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Title', 'Deck', 'FamilySize', 'IsAlone', 'FarePerPerson']
target = 'Survived'

# Xây dựng pipeline xử lý dữ liệu
# Phân loại các cột theo loại dữ liệu
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'FarePerPerson']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'IsAlone']

# Tạo preprocessor với ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Xây dựng pipeline hoàn chỉnh
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Chuẩn bị dữ liệu cho mô hình
X_train = train_processed[features]
y_train = train_processed[target]
X_test = test_processed[features]

# Tối ưu hóa siêu tham số với GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10]
}

# Chỉ sử dụng một tập con của dữ liệu để tìm kiếm siêu tham số nhanh hơn
X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=0.3, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_subset, y_subset)

print("\nTham số tốt nhất:", grid_search.best_params_)
print("Độ chính xác tốt nhất trên tập validation:", grid_search.best_score_)

# Huấn luyện mô hình với tham số tốt nhất
best_pipeline = grid_search.best_estimator_
best_pipeline.fit(X_train, y_train)

# Đánh giá mô hình bằng cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Các chỉ số đánh giá
accuracy_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
precision_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='precision')
recall_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='recall')
f1_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='f1')
roc_auc_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')

print("\nKết quả đánh giá cross-validation (5-fold):")
print(f"Accuracy: {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f}")
print(f"Precision: {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
print(f"Recall: {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
print(f"F1-score: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print(f"ROC-AUC: {roc_auc_scores.mean():.4f} ± {roc_auc_scores.std():.4f}")

# Trực quan hóa feature importance
rf_classifier = best_pipeline.named_steps['classifier']
preprocessor = best_pipeline.named_steps['preprocessor']

# Lấy tên các đặc trưng sau khi preprocessing
ohe = preprocessor.named_transformers_['cat']
cat_features = list(ohe.get_feature_names_out(categorical_features))
feature_names = numerical_features + cat_features

# Vẽ biểu đồ feature importance
plt.figure(figsize=(12, 8))
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# Lấy 10 đặc trưng quan trọng nhất
top_features = [feature_names[i] for i in indices[:10]]
print("\n10 đặc trưng quan trọng nhất:", top_features)

# Dự đoán trên tập test
test_predictions = best_pipeline.predict(X_test)

# Tạo file kết quả theo định dạng của Kaggle
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nĐã tạo file kết quả submission.csv")

# Tóm tắt quá trình
print("\nTóm tắt quá trình xử lý dữ liệu và xây dựng mô hình:")
print("1. Trích xuất đặc trưng từ tên (Title)")
print("2. Trích xuất đặc trưng từ cabin (Deck)")
print("3. Tạo đặc trưng gia đình (FamilySize, IsAlone, FarePerPerson)")
print("4. Dùng Linear Regression để điền giá trị Age bị thiếu")
print("5. Dùng phương pháp thay thế để điền giá trị Embarked và Fare bị thiếu")
print("6. Xây dựng pipeline xử lý dữ liệu và mô hình Random Forest")
print("7. Tối ưu hóa siêu tham số với GridSearchCV")
print("8. Đánh giá mô hình bằng cross-validation (5-fold)")
print("9. Phân tích feature importance")
print("10. Dự đoán và tạo file kết quả")