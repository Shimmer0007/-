import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_ind

def analyze_single_macro_risks(base_path, factors_path):
    """
    【最终修正版】加载因子收益和宏观意外数据，计算宏观利差。
    修正了合并前日期标准不统一的问题。
    """
    print("### 开始第四阶段-步骤B: 分析单一宏观风险暴露 ###")

    # 1. 定义文件路径
    path_shocks = os.path.join(base_path, 'macro_shocks.csv')
    path_factors = os.path.join(factors_path, 'all_factors_monthly.csv')
    output_path = os.path.join(base_path, 'exhibit3_replication_macro_spreads.csv')

    # 2. 加载并合并数据
    print("加载宏观意外和因子收益数据...")
    df_shocks = pd.read_csv(path_shocks)
    df_shocks['date'] = pd.to_datetime(df_shocks['date'])

    df_factors = pd.read_csv(path_factors)
    df_factors['date'] = pd.to_datetime(df_factors['date'])
    
    # 【修正】在合并前，将两个DataFrame的日期都标准化为月末日期
    df_shocks['date'] = df_shocks['date'] + pd.offsets.MonthEnd(0)
    df_factors['date'] = df_factors['date'] + pd.offsets.MonthEnd(0)
    
    # 使用inner join确保时间对齐
    df = pd.merge(df_factors, df_shocks, on='date', how='inner')
    print(f"数据成功合并，共有 {len(df)} 个月的重合数据用于分析。")

    # 3. 计算宏观利差 (Macro Spreads)
    factor_cols = ['SMB', 'HML', 'RMW', 'CMA', 'UMD']
    shock_cols = [col for col in df_shocks.columns if col != 'date']
    
    results = []

    print("正在计算宏观利差...")
    for shock_col in shock_cols:
        q1_bound = df[shock_col].quantile(0.25)
        q4_bound = df[shock_col].quantile(0.75)
        
        df_low_shock = df[df[shock_col] <= q1_bound]
        df_high_shock = df[df[shock_col] >= q4_bound]
        
        for factor_col in factor_cols:
            returns_low = df_low_shock[factor_col]
            returns_high = df_high_shock[factor_col]
            
            mean_low = returns_low.mean() * 12
            mean_high = returns_high.mean() * 12
            
            macro_spread = mean_high - mean_low
            
            t_stat, p_value = ttest_ind(returns_high, returns_low, equal_var=False, nan_policy='omit')
            
            results.append({
                'Macro Shock': shock_col.replace('_shock_ortho', ''),
                'Factor': factor_col,
                'Macro Spread (Annualized)': macro_spread,
                'P-value': p_value
            })

    # 4. 整理并保存结果
    df_results = pd.DataFrame(results)
    
    df_pivot = df_results.pivot_table(index='Macro Shock', columns='Factor', values='Macro Spread (Annualized)')
    df_p_values = df_results.pivot_table(index='Macro Shock', columns='Factor', values='P-value')
    
    def add_stars(val, p_val):
        if p_val < 0.01: return f"{val:.4f}***"
        elif p_val < 0.05: return f"{val:.4f}**"
        elif p_val < 0.1: return f"{val:.4f}*"
        else: return f"{val:.4f}"

    df_display = df_pivot.copy()
    for factor in df_display.columns:
        for shock in df_display.index:
            p_val = df_p_values.loc[shock, factor]
            val = df_pivot.loc[shock, factor]
            df_display.loc[shock, factor] = add_stars(val, p_val)

    final_factor_order = ['SMB', 'HML', 'UMD', 'RMW', 'CMA']
    final_factor_order = [col for col in final_factor_order if col in df_display.columns]
    df_display = df_display[final_factor_order]

    df_display.to_csv(output_path)
    
    print(f"\n### 单一宏观风险分析完成！ ###")
    print(f"结果已保存至: {output_path}")
    print("\nA股因子对宏观意外的敏感度 (宏观利差):")
    print(df_display)
    print("\n注: *, **, *** 分别代表在10%, 5%, 1%的水平上显著。")


if __name__ == '__main__':
    macro_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/mae'
    factors_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/factor'
    
    analyze_single_macro_risks(macro_base_path, factors_base_path)