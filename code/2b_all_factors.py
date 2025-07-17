import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def calculate_all_factors(base_path):
    """
    加载所有必需数据，计算Fama-French五因子(SMB, HML, RMW, CMA)及动量因子(UMD)。
    """
    print("### Phase 1: 数据准备与排序变量计算 ###")

    # 1. 定义文件路径
    path_integrated = os.path.join(base_path, 'factor/integrated_monthly_data.csv')
    path_combas = os.path.join(base_path, 'factor/账面价值数据FS_Combas/FS_Combas.csv')
    path_comins = os.path.join(base_path, 'factor/利润表FS_Comins/FS_Comins.csv')
    output_path = os.path.join(base_path, 'factor/all_factors_monthly.csv')

    # 2. 加载数据
    print("加载已整合的月度数据...")
    df = pd.read_csv(path_integrated)
    df['date'] = pd.to_datetime(df['date'])

    print("加载资产负债表 (FS_Combas)...")
    df_c = pd.read_csv(path_combas)
    df_c = df_c[df_c['Typrep'] == 'A'] # 筛选合并报表
    df_c = df_c[['Stkcd', 'Accper', 'A001000000', 'A003000000']]
    df_c.rename(columns={'Stkcd': 'stock_code', 'Accper': 'acc_date', 
                         'A001000000': 'TA', 'A003000000': 'BE'}, inplace=True)
    df_c['acc_date'] = pd.to_datetime(df_c['acc_date'])
    df_c = df_c[df_c['acc_date'].dt.month == 12] # 只保留年报

    print("加载利润表 (FS_Comins)...")
    df_i = pd.read_csv(path_comins)
    df_i = df_i[df_i['Typrep'] == 'A']
    df_i = df_i[['Stkcd', 'Accper', 'B001300000']]
    df_i.rename(columns={'Stkcd': 'stock_code', 'Accper': 'acc_date', 'B001300000': 'OP'}, inplace=True)
    df_i['acc_date'] = pd.to_datetime(df_i['acc_date'])
    df_i = df_i[df_i['acc_date'].dt.month == 12]

    # 3. 计算新的排序变量
    print("计算盈利能力(OP/BE)和资产增长率(AG)...")
    df_financials = pd.merge(df_c, df_i, on=['stock_code', 'acc_date'], how='inner')
    df_financials.dropna(subset=['TA', 'BE', 'OP'], inplace=True)
    df_financials = df_financials[(df_financials['BE'] > 0) & (df_financials['TA'] > 0)]
    
    # 计算盈利能力 (Operating Profitability)
    df_financials['op_be'] = df_financials['OP'] / df_financials['BE']
    
    # 计算资产增长率 (Asset Growth)
    df_financials.sort_values(by=['stock_code', 'acc_date'], inplace=True)
    df_financials['TA_last_year'] = df_financials.groupby('stock_code')['TA'].shift(1)
    df_financials['asset_growth'] = (df_financials['TA'] / df_financials['TA_last_year']) - 1
    
    df_financials['year'] = df_financials['acc_date'].dt.year
    df_financials = df_financials[['stock_code', 'year', 'op_be', 'asset_growth']]
    df_financials.dropna(subset=['op_be', 'asset_growth'], inplace=True)

    print("计算动量(Momentum)...")
    df.sort_values(by=['stock_code', 'date'], inplace=True)
    # 计算 t-12 到 t-2 的11个月的累计收益
    df['ret_plus_1'] = 1 + df['ret']
    # 使用 rolling.apply 需要一个 engine='cython' 不支持的函数，可能会慢
    # 一个更快的向量化方法是计算对数收益率的滚动和
    df['log_ret'] = np.log(df['ret_plus_1'])
    # 滚动11个月求和，然后shift(2)来定位到 t-2
    df['momentum'] = df.groupby('stock_code')['log_ret'].rolling(window=11, min_periods=11).sum().reset_index(0,drop=True).shift(2)
    df.drop(columns=['ret_plus_1', 'log_ret'], inplace=True)


    print("将所有排序变量合并到主数据表...")
    # 使用之前创建的linking_year来合并年度财务数据
    if 'linking_year' not in df.columns:
         df['linking_year'] = np.where(df['date'].dt.month >= 7, df['date'].dt.year, df['date'].dt.year - 1)
    
    df_financials.rename(columns={'year': 'linking_year'}, inplace=True)
    df = pd.merge(df, df_financials, on=['stock_code', 'linking_year'], how='left')

    print("\n### Phase 2: 年度排序与因子计算 ###")
    formation_years = sorted(df[df['date'].dt.month == 6]['date'].dt.year.unique())
    all_factors_data = []

    for year in tqdm(formation_years, desc="年度组合构建进度"):
        form_data = df[(df['date'].dt.year == year) & (df['date'].dt.month == 6)].copy()
        form_data.dropna(subset=['mv', 'bm_ratio', 'momentum', 'op_be', 'asset_growth'], inplace=True)

        if form_data.empty:
            print(f"警告：在 {year} 年6月没有足够数据，跳过。")
            continue

        # --- 排序和分组 ---
        # 规模
        size_break = form_data['mv'].median()
        form_data['size_group'] = np.where(form_data['mv'] <= size_break, 'S', 'B')
        # 价值(HML)
        bm_breaks = form_data['bm_ratio'].quantile([0.3, 0.7])
        form_data['bm_group'] = np.where(form_data['bm_ratio'] <= bm_breaks.iloc[0], 'L', np.where(form_data['bm_ratio'] <= bm_breaks.iloc[1], 'M', 'H'))
        # 盈利(RMW)
        op_be_breaks = form_data['op_be'].quantile([0.3, 0.7])
        form_data['op_be_group'] = np.where(form_data['op_be'] <= op_be_breaks.iloc[0], 'W', np.where(form_data['op_be'] <= op_be_breaks.iloc[1], 'M', 'R'))
        # 投资(CMA)
        ag_breaks = form_data['asset_growth'].quantile([0.3, 0.7])
        form_data['ag_group'] = np.where(form_data['asset_growth'] <= ag_breaks.iloc[0], 'C', np.where(form_data['asset_growth'] <= ag_breaks.iloc[1], 'M', 'A'))
        # 动量(UMD)
        mom_breaks = form_data['momentum'].quantile([0.3, 0.7])
        form_data['mom_group'] = np.where(form_data['momentum'] <= mom_breaks.iloc[0], 'D', np.where(form_data['momentum'] <= mom_breaks.iloc[1], 'M', 'U'))

        # 创建投资组合标签
        form_data['p_hml'] = form_data['size_group'] + '/' + form_data['bm_group']
        form_data['p_rmw'] = form_data['size_group'] + '/' + form_data['op_be_group']
        form_data['p_cma'] = form_data['size_group'] + '/' + form_data['ag_group']
        
        portfolio_map = form_data[['stock_code', 'p_hml', 'p_rmw', 'p_cma', 'mom_group']]

        # --- 计算持有期收益 ---
        start_year, start_month = (year, 7)
        end_year, end_month = (year + 1, 6)
        holding_period = df[((df['date'].dt.year == start_year) & (df['date'].dt.month >= start_month)) |
                             ((df['date'].dt.year == end_year) & (df['date'].dt.month <= end_month))].copy()
        
        holding_period = pd.merge(holding_period, portfolio_map, on='stock_code', how='inner')
        if holding_period.empty: continue

        def vw_ret(group): return (group['ret'] * group['mv']).sum() / group['mv'].sum()

        ret_hml = holding_period.groupby(['date', 'p_hml']).apply(vw_ret).unstack()
        ret_rmw = holding_period.groupby(['date', 'p_rmw']).apply(vw_ret).unstack()
        ret_cma = holding_period.groupby(['date', 'p_cma']).apply(vw_ret).unstack()
        ret_umd = holding_period.groupby(['date', 'mom_group']).apply(vw_ret).unstack()

        # --- 计算因子 ---
        smb_hml = ((ret_hml['S/L']+ret_hml['S/M']+ret_hml['S/H'])/3 - (ret_hml['B/L']+ret_hml['B/M']+ret_hml['B/H'])/3)
        smb_rmw = ((ret_rmw['S/W']+ret_rmw['S/M']+ret_rmw['S/R'])/3 - (ret_rmw['B/W']+ret_rmw['B/M']+ret_rmw['B/R'])/3)
        smb_cma = ((ret_cma['S/A']+ret_cma['S/M']+ret_cma['S/C'])/3 - (ret_cma['B/A']+ret_cma['B/M']+ret_cma['B/C'])/3)
        
        SMB = (smb_hml + smb_rmw + smb_cma) / 3
        HML = ((ret_hml['S/H'] + ret_hml['B/H'])/2 - (ret_hml['S/L'] + ret_hml['B/L'])/2)
        RMW = ((ret_rmw['S/R'] + ret_rmw['B/R'])/2 - (ret_rmw['S/W'] + ret_rmw['B/W'])/2)
        CMA = ((ret_cma['S/C'] + ret_cma['B/C'])/2 - (ret_cma['S/A'] + ret_cma['B/A'])/2)
        UMD = ret_umd['U'] - ret_umd['D']

        factors = pd.DataFrame({'SMB':SMB, 'HML':HML, 'RMW':RMW, 'CMA':CMA, 'UMD':UMD})
        all_factors_data.append(factors)

    if all_factors_data:
        final_factors_df = pd.concat(all_factors_data).sort_index()
        final_factors_df.to_csv(output_path)
        print("\n### Phase 3: 因子计算完成！ ###")
        print(f"所有因子收益率序列已保存至: {output_path}")
        print("因子数据预览:")
        print(final_factors_df.head())
    else:
        print("\n最终结果为空。请仔细检查输入数据。")

    return final_factors_df

if __name__ == '__main__':
    project_base_path = '/home/shimmer/projects/20250716_factor_investing/data'
    calculate_all_factors(project_base_path)