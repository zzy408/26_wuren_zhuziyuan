# 任务1：多元线性回归 —— 加州房价预测
# 自定义多元线性回归类（基于正规方程），对 California Housing 数据集进行建模与评估。

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 加载数据
data_path = "/home/zhuziyuan/learn/搜索算法与机器学习/homework/作业2/CaliforniaHousingPredict/CaliforniaHousing/cal_housing.data"
data = np.loadtxt(data_path, delimiter=",")

X = data[:, :-1]  # 前8列为特征
y = data[:, -1]   # 最后一列为目标房价

feature_names = ["longitude", "latitude", "housingMedianAge", "totalRooms",
                 "totalBedrooms", "population", "households", "medianIncome"]

print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"特征名: {feature_names}")
print(f"目标值范围: {y.min():.2f} ~ {y.max():.2f}")

# 可视化每个特征与目标的关系
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, name in enumerate(feature_names):
    axes[i].scatter(X[:, i], y, s=2, alpha=0.3)
    axes[i].set_xlabel(name)
    axes[i].set_ylabel("medianHouseValue")
plt.tight_layout()
plt.show()

# 自定义多元线性回归类
class MyLinearRegression:
    """自定义多元线性回归：使用正规方程（Normal Equation）求解。
    y = X @ w + b
    正规方程: w = (XᵀX)⁻¹ Xᵀy
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])

        theta = np.linalg.inv(X.T @ X) @ X.T @ y

        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

# 标准化：对特征做 Z-score 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集均值: {X_train_scaled.mean(axis=0).round(4)}")
print(f"训练集标准差: {X_train_scaled.std(axis=0).round(4)}")

# 训练自定义线性回归
model = MyLinearRegression(fit_intercept=True)
model.fit(X_train_scaled, y_train)

print("=== 训练结果 ===")
print(f"截距 (intercept): {model.intercept_:.4f}")
print(f"权重系数 (coef):")
for name, w in zip(feature_names, model.coef_):
    print(f"  {name:20s}: {w:12.4f}")

# 预测与评估
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_r2 = model.score(X_train_scaled, y_train)
test_r2 = model.score(X_test_scaled, y_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("========== 多元线性回归（无正则化）评估 ==========")
print(f"{'指标':<12} {'训练集':>14} {'测试集':>14}")
print("-" * 40)
print(f"{'R²':<12} {train_r2:14.4f} {test_r2:14.4f}")
print(f"{'MSE':<12} {train_mse:14.2f} {test_mse:14.2f}")
print(f"{'RMSE':<12} {np.sqrt(train_mse):14.2f} {np.sqrt(test_mse):14.2f}")
print(f"{'MAE':<12} {train_mae:14.2f} {test_mae:14.2f}")

# 绘制预测值与真实值的散点图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_true, y_pred, title in [
    (axes[0], y_train, y_train_pred, "训练集"),
    (axes[1], y_test, y_test_pred, "测试集")
]:
    ax.scatter(y_true, y_pred, s=5, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            'r--', linewidth=1.5, label="y = x")
    ax.set_xlabel("真实房价")
    ax.set_ylabel("预测房价")
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.show()

# 预测误差分布
errors = y_test - y_test_pred
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(errors, bins=50, edgecolor="k", alpha=0.7)
axes[0].axvline(0, color='r', linestyle='--', linewidth=1.5)
axes[0].set_xlabel("预测误差")
axes[0].set_ylabel("频数")
axes[0].set_title("测试集预测误差分布")

axes[1].scatter(y_test_pred, errors, s=5, alpha=0.5)
axes[1].axhline(0, color='r', linestyle='--', linewidth=1.5)
axes[1].set_xlabel("预测值")
axes[1].set_ylabel("误差")
axes[1].set_title("残差图")

plt.tight_layout()
plt.show()

print(f"误差均值: {errors.mean():.2f}")
print(f"误差标准差: {errors.std():.2f}")
print(f"误差偏度: {np.mean(errors**3) / (errors.std()**3 + 1e-10):.4f}")
