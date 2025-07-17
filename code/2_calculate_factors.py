import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def calculate_ff_factors(data_path, output_path):
    """
    加载整合后的月度数据，计算Fama-French SMB和HML因子。
    V2: 修正了日期匹配逻辑，使其更加稳健。
    """
    print("开始计算Fama-French因子 (SMB, HML)...")

    # 1. 加载整合好的数据
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    df['mv'] = pd.to_numeric(df['mv'], errors='coerce')
    df['bm_ratio'] = pd.to_numeric(df['bm_ratio'], errors='coerce')
    df.dropna(subset=['mv', 'bm_ratio'], inplace=True)

    # 2. 准备年度循环
    formation_years = sorted(df[df['date'].dt.month == 6]['date'].dt.year.unique())
    all_factors_data = []

    print(f"将从 {formation_years[0]} 年开始，到 {formation_years[-1]} 年结束，逐年构建投资组合...")

    for year in tqdm(formation_years, desc="年度组合构建进度"):
        # --- 3. 每年6月进行组合构建 ---
        # 【修正】使用年份和月份进行筛选，而不是具体的某一天
        form_data = df[(df['date'].dt.year == year) & (df['date'].dt.month == 6)].copy()
        
        # 【增加】如果当年6月没有数据，则跳过
        if form_data.empty:
            print(f"警告：在 {year} 年6月没有找到任何数据，跳过这一年。")
            continue

        form_data.dropna(subset=['mv', 'bm_ratio'], inplace=True)
        if form_data.empty: # 再次检查，以防dropna后变空
            print(f"警告：在 {year} 年6月的数据dropna后为空，跳过这一年。")
            continue

        size_breakpoint = form_data['mv'].median()
        bm_breakpoint_low = form_data['bm_ratio'].quantile(0.3)
        bm_breakpoint_high = form_data['bm_ratio'].quantile(0.7)

        form_data['size_group'] = np.where(form_data['mv'] <= size_breakpoint, 'S', 'B')
        form_data['bm_group'] = np.where(form_data['bm_ratio'] <= bm_breakpoint_low, 'L',
                                       np.where(form_data['bm_ratio'] <= bm_breakpoint_high, 'M', 'H'))
        
        form_data['portfolio'] = form_data['size_group'] + '/' + form_data['bm_group']
        portfolio_map = form_data[['stock_code', 'portfolio']]

        # --- 4. 计算接下来12个月的组合收益率 ---
        start_year, start_month_num = (year, 7)
        end_year, end_month_num = (year + 1, 6)

        # 【修正】使用更稳健的逻辑来筛选12个月的持有期
        holding_period_data = df[
            ((df['date'].dt.year == start_year) & (df['date'].dt.month >= start_month_num)) |
            ((df['date'].dt.year == end_year) & (df['date'].dt.month <= end_month_num))
        ].copy()
        
        holding_period_data = pd.merge(holding_period_data, portfolio_map, on='stock_code', how='inner')
        
        if holding_period_data.empty:
            print(f"警告：在 {year}-{year+1} 持有期内，没有匹配到任何股票，跳过。")
            continue

        def value_weighted_return(group):
            weights = group['mv'] / group['mv'].sum()
            return (group['ret'] * weights).sum()

        monthly_returns = holding_period_data.groupby(['date', 'portfolio']).apply(value_weighted_return).unstack()

        expected_portfolios = ['S/L', 'S/M', 'S/H', 'B/L', 'B/M', 'B/H']
        for p in expected_portfolios:
            if p not in monthly_returns.columns:
                monthly_returns[p] = 0
        
        smb = (monthly_returns['S/L'] + monthly_returns['S/M'] + monthly_returns['S/H']) / 3 - \
              (monthly_returns['B/L'] + monthly_returns['B/M'] + monthly_returns['B/H']) / 3
        
        hml = (monthly_returns['S/H'] + monthly_returns['B/H']) / 2 - \
              (monthly_returns['S/L'] + monthly_returns['B/L']) / 2
        
        factors_this_year = pd.DataFrame({'SMB': smb, 'HML': hml})
        all_factors_data.append(factors_this_year)

    if all_factors_data:
        final_factors_df = pd.concat(all_factors_data)
        final_factors_df.sort_index(inplace=True)
        final_factors_df.to_csv(output_path)
        
        print("\n因子计算完成！")
        print(f"因子收益率序列已保存至: {output_path}")
        print("因子数据预览:")
        print(final_factors_df.head())
    else:
        print("\n最终结果仍为空。请检查输入数据中是否包含足够年份的6月份数据。")

    return final_factors_df

if __name__ == '__main__':
    input_data_path = '/home/shimmer/projects/20250716_factor_investing/data/factor/integrated_monthly_data.csv'
    output_factors_path = '/home/shimmer/projects/20250716_factor_investing/data/factor/fama_french_factors.csv'
    
    calculate_ff_factors(input_data_path, output_factors_path)