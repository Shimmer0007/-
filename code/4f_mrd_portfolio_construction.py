import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from arch import arch_model

def mrd_portfolio_construction_final(base_path, factors_path):
    """
    【最终修正版】构建MRD组合。修正了优化器目标函数的数值尺度问题。
    """
    print("### 开始最终任务: 构建宏观风险最小化(MRD)投资组合 (V2) ###")

    # --- 1. 数据准备 (与之前脚本逻辑相同) ---
    print("加载并准备所有分析所需数据...")
    df = prepare_full_dataset(base_path, factors_path) # 使用之前的辅助函数
    factor_cols = ['SMB', 'HML', 'UMD', 'RMW', 'CMA']
    regime_cols = ['risk_tolerance', 'macro_outlook', 'macro_stability', 'risk_on_off']
    
    # --- 2. 定义经过优化的优化引擎 ---
    def build_mrd_portfolio(df, factor_cols, target_regimes):
        n_factors = len(factor_cols)
        uncond_means = df[factor_cols].mean()
        
        cond_means = []
        for regime in target_regimes:
            if regime != 'risk_on_off':
                q1 = df[regime].quantile(0.25)
                q4 = df[regime].quantile(0.75)
                states = [df[regime] <= q1, df[regime] >= q4]
            else:
                states = [df[regime] == -1, df[regime] == 1]
            for state in states:
                cond_means.append(df[state][factor_cols].mean())

        def objective_func(weights):
            # 【修正】将所有收益率年化(乘以12)，以放大目标函数的尺度
            portfolio_uncond_mean = np.dot(weights, uncond_means) * 12
            deviations = []
            for cm in cond_means:
                portfolio_cond_mean = np.dot(weights, cm) * 12
                deviations.append((portfolio_cond_mean - portfolio_uncond_mean)**2)
            return np.sum(deviations)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n_factors))
        initial_weights = np.array([1/n_factors] * n_factors)
        
        opt_result = minimize(objective_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return opt_result.x

    # --- 3. 构建不同的投资组合 ---
    # ... (后续代码与之前版本完全相同) ...
    print("正在构建等权组合和不同的MRD组合...")
    ew_weights = np.array([1/len(factor_cols)] * len(factor_cols))
    df['ew_ret'] = df[factor_cols].dot(ew_weights)
    
    mrd_mo_weights = build_mrd_portfolio(df, factor_cols, ['macro_outlook'])
    df['mrd_mo_ret'] = df[factor_cols].dot(mrd_mo_weights)
    
    mrd_all_weights = build_mrd_portfolio(df, factor_cols, regime_cols)
    df['mrd_all_ret'] = df[factor_cols].dot(mrd_all_weights)
    
    print("正在分析和对比各组合表现...")
    portfolios = {'EW': 'ew_ret', 'MRD_Macro_Outlook': 'mrd_mo_ret', 'MRD_All_Regimes': 'mrd_all_ret'}
    analysis_results = []
    for name, ret_col in portfolios.items():
        annual_ret = df[ret_col].mean() * 12
        annual_vol = df[ret_col].std() * np.sqrt(12)
        sharpe_ratio = annual_ret / annual_vol if annual_vol != 0 else 0
        
        q1_mo = df['macro_outlook'].quantile(0.25)
        q4_mo = df['macro_outlook'].quantile(0.75)
        ret_good = df[df['macro_outlook'] >= q4_mo][ret_col].mean() * 12
        ret_bad = df[df['macro_outlook'] <= q1_mo][ret_col].mean() * 12
        spread = ret_good - ret_bad
        
        analysis_results.append({
            'Portfolio': name, 'Annual Return': annual_ret, 'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio, 'Macro Outlook Spread': spread
        })
    df_display = pd.DataFrame(analysis_results).set_index('Portfolio')
    
    print("\n### 最终组合构建与分析完成！ ###")
    print("\n不同因子组合的无条件与条件表现对比:")
    print(df_display.to_string(float_format="%.4f"))
    print("\nMRD组合权重分配:")
    print("MRD on Macro Outlook:", dict(zip(factor_cols, mrd_mo_weights)))
    print("MRD on All Regimes:  ", dict(zip(factor_cols, mrd_all_weights)))

def prepare_full_dataset(base_path, factors_path):
    """一个辅助函数，整合了之前所有数据准备步骤"""
    # (此处省略了之前4c脚本中加载和合并所有数据的完整代码，仅为示意)
    # 实际运行时，此部分代码应与4e脚本中的数据准备部分相同
    path_factors=os.path.join(factors_path,'all_factors_monthly.csv');df_factors=pd.read_csv(path_factors);df_factors['date']=pd.to_datetime(df_factors['date'])+pd.offsets.MonthEnd(0)
    path_shocks=os.path.join(base_path,'macro_shocks.csv');df_shocks=pd.read_csv(path_shocks);df_shocks['date']=pd.to_datetime(df_shocks['date'])+pd.offsets.MonthEnd(0)
    path_ip_growth=os.path.join(base_path,'月度工业增加值/CME_Mindust1.csv');df_ip=pd.read_csv(path_ip_growth);df_ip=df_ip[(df_ip['Datatype']==2)&(df_ip['Datasgn']=='A')&(df_ip['Fresgn']=='M')];df_ip=df_ip[['Staper','Eindm0101']];df_ip.rename(columns={'Staper':'date','Eindm0101':'ip_growth'},inplace=True);df_ip['date']=pd.to_datetime(df_ip['date'])+pd.offsets.MonthEnd(0);df_ip['ip_growth']/=100
    path_rf=os.path.join(base_path,'macro_variables_part1.csv');df_rf=pd.read_csv(path_rf)[['date','short_rate']];df_rf['date']=pd.to_datetime(df_rf['date'])+pd.offsets.MonthEnd(0)
    path_index=os.path.join(base_path,'指数文件/TRD_Index.csv');df_mkt=pd.read_csv(path_index);df_mkt=df_mkt[df_mkt['Indexcd']==300];df_mkt['date']=pd.to_datetime(df_mkt['Trddt']);mkt_ret=(1+df_mkt.set_index('date')['Retindex']).resample('ME').prod()-1;mkt_ret.name='mkt_ret'
    df=pd.merge(df_factors,df_shocks,on='date');df=pd.merge(df,df_ip,on='date');df=pd.merge(df,pd.DataFrame(mkt_ret),on='date');df=pd.merge(df,df_rf,on='date');df['mkt_excess_ret']=df['mkt_ret']-df['short_rate'];df.set_index('date',inplace=True);df.sort_index(inplace=True);df.dropna(inplace=True)
    shock_cols=[col for col in df_shocks.columns if '_shock_ortho' in col];X_shocks=df[shock_cols]
    y_rt=df['mkt_excess_ret'].shift(-12);df_rt=pd.concat([y_rt,X_shocks],axis=1).dropna();reg_rt=LinearRegression().fit(df_rt[shock_cols],df_rt['mkt_excess_ret']);df['risk_tolerance']=reg_rt.predict(X_shocks)
    y_mo=df['ip_growth'].shift(-12);df_mo=pd.concat([y_mo,X_shocks],axis=1).dropna();reg_mo=LinearRegression().fit(df_mo[shock_cols],df_mo['ip_growth']);df['macro_outlook']=reg_mo.predict(X_shocks)
    shocks_for_garch=[col for col in shock_cols if 'market_vol' not in col];cond_vols=pd.DataFrame(index=X_shocks.index)
    for shock in shocks_for_garch:garch_model=arch_model(X_shocks[shock],vol='Garch',p=1,q=1);res=garch_model.fit(disp='off');cond_vols[shock]=res.conditional_volatility
    uncertainty=PCA(n_components=1).fit_transform(cond_vols);df['macro_stability']=-uncertainty
    df['risk_on_off']=0;df.loc[(df['market_ep_shock_ortho']<0)&(df['market_vol_shock_ortho']<0),'risk_on_off']=1;df.loc[(df['market_ep_shock_ortho']>0)&(df['market_vol_shock_ortho']>0),'risk_on_off']=-1
    return df.reset_index()


if __name__ == '__main__':
    macro_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/mae'
    factors_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/factor'
    mrd_portfolio_construction_final(macro_base_path, factors_base_path)