import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from arch import arch_model

def analyze_diversification(base_path, factors_path):
    """
    计算并对比标准收益率相关性和宏观风险暴露相关性。
    复现原文Exhibit 7的核心逻辑。
    """
    print("### 开始第四阶段-步骤D: 分析因子间的伪分散效应 ###")

    # 1. 加载并准备所有需要的数据
    # 这个过程与上一个脚本类似，我们需要重新构建情景指标
    print("加载并准备所有分析所需数据...")
    path_factors = os.path.join(factors_path, 'all_factors_monthly.csv')
    path_shocks = os.path.join(base_path, 'macro_shocks.csv')
    path_ip_growth = os.path.join(base_path, '月度工业增加值/CME_Mindust1.csv')
    path_index = os.path.join(base_path, '指数文件/TRD_Index.csv')
    path_rf = os.path.join(base_path, 'macro_variables_part1.csv')

    df_factors = pd.read_csv(path_factors)
    df_factors['date'] = pd.to_datetime(df_factors['date']) + pd.offsets.MonthEnd(0)

    df_shocks = pd.read_csv(path_shocks)
    df_shocks['date'] = pd.to_datetime(df_shocks['date']) + pd.offsets.MonthEnd(0)

    # 准备工业增加值
    df_ip = pd.read_csv(path_ip_growth)
    df_ip = df_ip[(df_ip['Datatype'] == 2) & (df_ip['Datasgn'] == 'A') & (df_ip['Fresgn'] == 'M')]
    df_ip = df_ip[['Staper', 'Eindm0101']]
    df_ip.rename(columns={'Staper': 'date', 'Eindm0101': 'ip_growth'}, inplace=True)
    df_ip['date'] = pd.to_datetime(df_ip['date']) + pd.offsets.MonthEnd(0)
    df_ip['ip_growth'] = df_ip['ip_growth'] / 100

    # 准备市场超额收益
    df_rf = pd.read_csv(path_rf)[['date', 'short_rate']]
    df_rf['date'] = pd.to_datetime(df_rf['date']) + pd.offsets.MonthEnd(0)
    
    df_mkt = pd.read_csv(path_index)
    df_mkt = df_mkt[df_mkt['Indexcd'] == 300]
    df_mkt['date'] = pd.to_datetime(df_mkt['Trddt'])
    mkt_ret = (1 + df_mkt.set_index('date')['Retindex']).resample('ME').prod() - 1
    mkt_ret.name = 'mkt_ret'

    # 合并成主DataFrame
    df = pd.merge(df_factors, df_shocks, on='date')
    df = pd.merge(df, df_ip, on='date')
    df = pd.merge(df, pd.DataFrame(mkt_ret), on='date')
    df = pd.merge(df, df_rf, on='date')
    df['mkt_excess_ret'] = df['mkt_ret'] - df['short_rate']
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    # --- 构建四个综合情景指标 (与4c脚本逻辑相同) ---
    shock_cols = [col for col in df_shocks.columns if '_shock_ortho' in col]
    X_shocks = df[shock_cols]
    
    # 风险容忍度
    y_rt = df['mkt_excess_ret'].shift(-12)
    df_rt = pd.concat([y_rt, X_shocks], axis=1).dropna()
    reg_rt = LinearRegression().fit(df_rt[shock_cols], df_rt['mkt_excess_ret'])
    df['risk_tolerance'] = reg_rt.predict(X_shocks)
    
    # 宏观展望
    y_mo = df['ip_growth'].shift(-12)
    df_mo = pd.concat([y_mo, X_shocks], axis=1).dropna()
    reg_mo = LinearRegression().fit(df_mo[shock_cols], df_mo['ip_growth'])
    df['macro_outlook'] = reg_mo.predict(X_shocks)
    
    # 宏观稳定性
    shocks_for_garch = [col for col in shock_cols if 'market_vol' not in col]
    cond_vols = pd.DataFrame(index=X_shocks.index)
    for shock in shocks_for_garch:
        garch_model = arch_model(X_shocks[shock], vol='Garch', p=1, q=1)
        res = garch_model.fit(disp='off')
        cond_vols[shock] = res.conditional_volatility
    uncertainty = PCA(n_components=1).fit_transform(cond_vols)
    df['macro_stability'] = -uncertainty
    
    # 风险偏好开关 (我们用数值表示以便计算beta)
    df['risk_on_off'] = 0
    df.loc[(df['market_ep_shock_ortho'] < 0) & (df['market_vol_shock_ortho'] < 0), 'risk_on_off'] = 1  # Risk-On
    df.loc[(df['market_ep_shock_ortho'] > 0) & (df['market_vol_shock_ortho'] > 0), 'risk_on_off'] = -1 # Risk-Off
    print("数据准备与情景指标构建完成。")

    # --- 2. 计算标准收益率相关性 ---
    print("\n--- A股因子标准收益率相关性矩阵 ---")
    factor_cols = ['SMB', 'HML', 'UMD', 'RMW', 'CMA']
    return_corr_matrix = df[factor_cols].corr()
    print(return_corr_matrix)

    # --- 3. 计算宏观风险暴露(Beta)及相关性 ---
    print("\n--- 计算各因子对宏观情景的暴露(Beta)... ---")
    regime_cols = ['risk_tolerance', 'macro_outlook', 'macro_stability', 'risk_on_off']
    X_regimes = sm.add_constant(df[regime_cols]) # 加入常数项
    
    macro_betas = pd.DataFrame()
    
    for factor in factor_cols:
        y_factor = df[factor]
        model = sm.OLS(y_factor, X_regimes).fit()
        macro_betas[factor] = model.params.drop('const') # 保存除常数项外的beta系数
        
    print("\n--- A股因子宏观风险暴露(Beta)相关性矩阵 ---")
    macro_beta_corr_matrix = macro_betas.corr()
    print(macro_beta_corr_matrix)

    print("\n### 伪分散效应分析完成！ ###")


if __name__ == '__main__':
    macro_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/mae'
    factors_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/factor'
    
    analyze_diversification(macro_base_path, factors_base_path)