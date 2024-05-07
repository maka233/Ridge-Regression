## 可能需要接着model代码后写

import matplotlib.pyplot as plt
import numpy as np 
import matplotlib
from sklearn import model_selection
from sklearn.linear_model import Ridge, RidgeCV

# 构造不同的Lambda值
ks = np.logspace(-5, 2, 200)
# 构造空列表，用于存储模型的偏回归系数
ridge_cofficients = []
# 循环迭代不同的k值
for k in ks:
    ridge = Ridge(alpha=k)
    ridge.fit(X_train, y_train)
    ridge_cofficients.append(ridge.coef_)
 
# 设置绘图风格
plt.style.use('ggplot')
plt.plot(ks, ridge_cofficients)
# 对x轴做对数处理
plt.xscale('log')
# 设置折线图x轴和y轴标签
plt.xlabel('Log(Lambda)')
plt.ylabel('Cofficients')
# 显示图形

# 设置交叉验证的参数，对于每一个Lambda值都执行10重交叉验证
ridge_cv = RidgeCV(alphas=ks, scoring='neg_mean_squared_error', cv=10)
# 模型拟合
ridge_cv.fit(X_train, y_train)
# 返回最佳的lambda
ridge_best_Lambda = ridge_cv.alpha_  # 0.013509935211980266
ridge_best_Lambda

# 基于最佳的Lambda值建模
ridge = Ridge(alpha=ridge_best_Lambda)
ridge.fit(X_train, y_train)
# 返回岭回归系数
res = pd.Series(index=['Intercept'] + X_train.columns.tolist(), data=[ridge.intercept_]+ridge.coef_.tolist())
print(res)

'''
# 查看回归系数  
coefficients = ridge.coef_  
coefficients
'''

from sklearn.metrics import r2_score 
# 预测训练集数据  
y_test_pred = ridge.predict(X_test)  
  
# 计算决定系数  
r2 = r2_score(y_test, y_test_pred)  
  
# 打印决定系数  
print(f"模型的决定系数 (R^2): {r2}")

## 是否通过检验？不太懂
from scipy.stats import f

n_features = X_train.shape[1]
dfn = n_features
dfd = X_train.shape[0] - n_features - 1
F = (ridge.score(X_train, y_train) / dfn) / ((1 - ridge.score(X_train, y_train)) / dfd)
p_value = 1 - f.cdf(F, dfn, dfd)

print("F值：", F)
print("p值：", p_value)

## # 导入第三方包中的函数
from sklearn.metrics import mean_squared_error
# 测试值与正确值做对比
ridge_predict = ridge.predict(X_test)
res = pd.DataFrame({
    'real': y_test,
    'pred': ridge_predict
})
RMSE = np.sqrt(mean_squared_error(y_test, ridge_predict))  # 0.1603073575150871  RMSE越小越好
# RMSE
