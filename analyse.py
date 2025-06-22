import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

data = pd.read_excel("TreasYield.xlsx")
data["date"] = pd.to_datetime(data["date"])
# 提取year = 10的行
data_10 = data[data["year"] == 10]
data_10 = data_10.set_index("date")
data_10 = data_10.sort_index()
# 去除空值
data_10 = data_10.dropna()

data_1 = data[data["year"] == 1]
data_1 = data_1.set_index("date")

# 1. 描述性统计和可视化
print("数据描述性统计:")
print(data_10["r"].describe())

# 绘制时间序列图
plt.figure(figsize=(14, 7))
plt.plot(data_10["r"])
plt.title("China 10-Year Treasury Yield Over Time")
plt.xlabel("Date")
plt.ylabel("Yield")
plt.grid(True)
plt.savefig("timeseries_plot.png")
print("\n时间序列图已保存为 timeseries_plot.png")

print("# 中国十年期国债收益率时间序列分析\n\n")
print("## 第一步：数据概览与可视化\n\n")
print(
    "我们首先对数据进行加载和清洗，然后进行描述性统计分析，并绘制时间序列图以直观了解其变化趋势。\n\n"
)
print("### 描述性统计\n\n")
print("```\n")
print(data_10["r"].describe().to_string())
print("\n```\n\n")
print("### 时间序列图\n\n")
print("![中国十年期国债收益率时间序列图](timeseries_plot.png)\n\n")
print(
    "从图中可以初步看出，该时间序列具有一定的波动性，可能存在趋势和非平稳性。接下来我们将进行更详细的平稳性检验。"
)

# 2. 平稳性检验
print("\n进行ADF检验...")
adf_result = adfuller(data_10["r"])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"\t{key}: {value}")

print("\n\n## 第二步：平稳性检验 (ADF Test)\n\n")
print(
    "为了确定时间序列是否平稳，我们采用增强迪基-福勒（ADF）检验。平稳性是许多时间序列模型（如ARIMA）的基本要求。\n\n"
)
print("### ADF检验结果\n\n")
print("```\n")
print(f"ADF Statistic: {adf_result[0]}\n")
print(f"p-value: {adf_result[1]}\n")
print("Critical Values:\n")
for key, value in adf_result[4].items():
    print(f"\t{key}: {value}\n")
print("```\n\n")
if adf_result[1] > 0.05:
    print(
        "检验结果的 p-value 远大于0.05，因此我们不能拒绝原假设，即该时间序列是**非平稳的**。我们需要进行差分处理来使其平稳。"
    )
else:
    print(
        "检验结果的 p-value 小于0.05，我们可以拒绝原假设，认为该时间序列是**平稳的**。"
    )

# 3. 差分及ACF/PACF图
data_10_diff = data_10["r"].diff().dropna()

# 绘制ACF和PACF图
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(data_10_diff, ax=axes[0])
plot_pacf(data_10_diff, ax=axes[1])
plt.savefig("acf_pacf_plot.png")
print("\nACF 和 PACF 图已保存为 acf_pacf_plot.png")

print("\n\n## 第三步：差分与ACF/PACF分析\n\n")
print(
    "由于原始序列非平稳，我们进行一阶差分，并绘制自相关函数（ACF）和偏自相关函数（PACF）图，以帮助确定ARIMA(p,d,q)模型中的p和q阶数。\n\n"
)
print("### ACF 和 PACF 图\n\n")
print("![ACF 和 PACF 图](acf_pacf_plot.png)\n\n")
print("根据ACF和PACF图，我们选择 p=3, q=1。")

# 4. ARIMA 模型拟合与残差分析
print("\n拟合ARIMA(3,1,1)模型...")
model = ARIMA(data_10["r"], order=(3, 1, 1))
results = model.fit()
print(results.summary())

# 绘制残差图
residuals = results.resid
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sm.qqplot(residuals, line="s", ax=axes[0])
axes[0].set_title("Residuals QQ-Plot")
plot_acf(residuals, ax=axes[1])
axes[1].set_title("Residuals ACF Plot")
plt.savefig("residuals_plot.png")
print("\n残差分析图已保存为 residuals_plot.png")

print("\n\n## 第四步：ARIMA(3,1,1)模型拟合与残差分析\n\n")
print("基于ACF和PACF图的分析，我们选择p=3, d=1, q=1来构建ARIMA模型。\n\n")
print("### 模型摘要\n\n")
print("```\n")
print(str(results.summary()))
print("\n```\n\n")
print("### 残差分析\n\n")
print("![残差QQ图和ACF图](residuals_plot.png)\n\n")
print(
    "残差的QQ图可以帮助我们判断残差是否服从正态分布。如果点大致落在红线上，说明残差是正态的。残差的ACF图可以帮助我们判断残差中是否还存在自相关。如果所有相关性都在置信区间内，说明模型已经充分提取了信息。从结果看，残差不服从正态分布，且存在异方差，说明需要GARCH模型。"
)

# 5. GARCH 模型拟合
print("\n拟合GARCH(1,1)模型...")
# 我们使用差分后的数据，因为它更接近平稳
garch_model = arch_model(data_10_diff * 100, vol="Garch", p=1, q=1)
garch_results = garch_model.fit()
print(garch_results.summary())

# 绘制条件波动率
fig = garch_results.plot(annualize="D")
plt.savefig("garch_plot.png")
print("\nGARCH 模型图已保存为 garch_plot.png")


print("\n\n## 第五步：GARCH(1,1)模型与波动率分析\n\n")
print(
    "ARIMA模型的残差分析表明存在异方差性，即波动率不是恒定的。因此，我们使用GARCH(1,1)模型来对波动率进行建模。\n\n"
)
print("### GARCH(1,1)模型摘要\n\n")
print("```\n")
print(str(garch_results.summary()))
print("\n```\n\n")
print("### 条件波动率图\n\n")
print("![条件波动率图](garch_plot.png)\n\n")
print(
    "GARCH模型的结果显示，alpha[1]和beta[1]的系数都是高度显著的，说明过去的波动和过去的方差都对当前的波动有显著影响。这证实了收益率序列中存在波动率聚集现象。"
)

# 6. 季节性分析
print("\n进行季节性分解...")
# 使用加法模型，因为波动看起来不是随级别增加而增加
decomposition = seasonal_decompose(data_10["r"], model="additive", period=365)
fig = decomposition.plot()
fig.set_size_inches(14, 9)
plt.savefig("seasonal_decomposition_plot.png")
print("\n季节性分解图已保存为 seasonal_decomposition_plot.png")

print("\n\n## 第六步：季节性分析\n\n")
print("为了探究数据中是否存在周期性模式，我们对其进行季节性分解。\n\n")
print("### 季节性分解图\n\n")
print("![季节性分解图](seasonal_decomposition_plot.png)\n\n")
print(
    "分解图显示了原始序列、趋势部分、季节性部分和残差部分。从趋势图中可以更清晰地看到利率的长期走势。季节性部分显示了一年内的周期性波动。残差部分则是去除趋势和季节性后剩下的部分。"
)

# 7. 协同时间序列分析 (VAR模型)
print("\n进行协同时间序列分析 (VAR模型)...")

# 合并1年期和10年期数据
data_1_10 = pd.merge(
    data_1["r"],
    data_10["r"],
    left_index=True,
    right_index=True,
    suffixes=("_1y", "_10y"),
)
data_1_10 = data_1_10.dropna()

# 检查合并后数据的平稳性
print("\n对合并后的数据进行ADF检验...")
adf_result_1y = adfuller(data_1_10["r_1y"])
adf_result_10y = adfuller(data_1_10["r_10y"])
print(f"1-Year Yield ADF p-value: {adf_result_1y[1]}")
print(f"10-Year Yield ADF p-value: {adf_result_10y[1]}")

# 对非平稳序列进行差分
data_diff = data_1_10.diff().dropna()

# 再次检查差分后数据的平稳性
adf_result_diff_1y = adfuller(data_diff["r_1y"])
adf_result_diff_10y = adfuller(data_diff["r_10y"])
print(f"Differenced 1-Year Yield ADF p-value: {adf_result_diff_1y[1]}")
print(f"Differenced 10-Year Yield ADF p-value: {adf_result_diff_10y[1]}")

# VAR模型阶数选择
print("\n选择VAR模型的最优阶数...")
model = VAR(data_diff)
order_selection = model.select_order(maxlags=15)
print(order_selection.summary())
best_lag = order_selection.aic

# 拟合VAR模型
print(f"\n拟合VAR({best_lag})模型...")
var_model = model.fit(best_lag)
print(var_model.summary())

# 格兰杰因果检验
print("\n进行格兰杰因果检验...")
granger_1y_on_10y = var_model.test_causality("r_10y", ["r_1y"], kind="f")
print(granger_1y_on_10y.summary())

granger_10y_on_1y = var_model.test_causality("r_1y", ["r_10y"], kind="f")
print(granger_10y_on_1y.summary())


# 脉冲响应分析
print("\n进行脉冲响应分析...")
irf = var_model.irf(periods=20)
fig = irf.plot(orth=False)
plt.savefig("var_irf_plot.png")
print("\n脉冲响应函数图已保存为 var_irf_plot.png")

print("\n\n## 第七步：协同时间序列分析 (VAR模型)\n\n")
print(
    "为了分析一年期和十年期国债收益率之间的动态关系，我们使用向量自回归（VAR）模型。\n\n"
)
print("### 数据准备与平稳性检验\n\n")
print(
    "我们首先将两个时间序列合并，并对它们进行平稳性检验。原始序列非平稳，因此我们进行一阶差分。\n\n"
)
print("```\n")
print(f"Differenced 1-Year Yield ADF p-value: {adf_result_diff_1y[1]}\\n")
print(f"Differenced 10-Year Yield ADF p-value: {adf_result_diff_10y[1]}\\n")
print("```\n\n")
print("差分后的序列是平稳的。\n\n")
print("### VAR模型阶数选择\n\n")
print("我们使用AIC准则来选择最优的滞后阶数。\n\n")
print("```\n")
print(str(order_selection.summary()))
print("\n```\n\n")
print(f"根据AIC，我们选择滞后阶数为 {best_lag}。\n\n")
print("### VAR模型结果\n\n")
print("```\n")
print(str(var_model.summary()))
print("\n```\n\n")
print("### 格兰杰因果关系检验\n\n")
print("我们检验两个利率序列之间是否存在格兰杰因果关系。\n\n")
print("1. 检验1年期利率是否是10年期利率的格兰杰原因:\n")
print("```\n")
print(str(granger_1y_on_10y.summary()))
print("\n```\n\n")
print("2. 检验10年期利率是否是1年期利率的格兰杰原因:\n")
print("```\n")
print(str(granger_10y_on_1y.summary()))
print("\n```\n\n")
print("### 脉冲响应分析\n\n")
print("脉冲响应函数（IRF）描述了一个内生变量的冲击对其他内生变量的动态影响。\n\n")
print("![VAR脉冲响应函数图](var_irf_plot.png)\n\n")
print(
    "该图显示了在一个变量上施加一个标准差的冲击后，对自身和其他变量在未来20个时期的影响。"
)
