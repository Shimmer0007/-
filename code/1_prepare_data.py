import pandas as pd
import os
import numpy as np

def integrate_csmar_data(base_path):
    """
    整合CSMAR的个股回报率、市值、账面价值和无风险利率数据，
    为构建Fama-French因子做准备。

    Args:
        base_path (str): 包含所有数据子文件夹的根目录路径。

    Returns:
        pandas.DataFrame: 一个包含所有已对齐和计算好的数据的DataFrame。
    """
    print("开始整合数据...")

    # 1. 定义文件路径
    path_ret = os.path.join(base_path, 'factor/个股月度回报率TRD_Mnth/TRD_Mnth.csv')
    path_mv = os.path.join(base_path, 'factor/市值数据TRD_Msmvttl/TRD_Msmvttl.csv') # 注意：文件名我根据文件夹名做了修正
    path_rf = os.path.join(base_path, 'factor/无风险利率/CME_Mfinamkt2.csv')
    path_be = os.path.join(base_path, 'factor/账面价值数据FS_Combas/FS_Combas.csv')
    
    output_path = os.path.join(base_path, 'factor/integrated_monthly_data.csv')

    # 2. 加载并处理各个数据集

    # --- 2.1 个股月度回报率 (Return) ---
    print("加载个股月度回报率...")
    df_ret = pd.read_csv(path_ret)
    df_ret = df_ret[['Stkcd', 'Trdmnt', 'Mretwd']]
    df_ret.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date', 'Mretwd': 'ret'}, inplace=True)
    df_ret['date'] = pd.to_datetime(df_ret['date'])
    # 删除回报率为空的记录
    df_ret.dropna(subset=['ret'], inplace=True)

    # --- 2.2 个股月度总市值 (Market Value) ---
    print("加载个股月度总市值...")
    df_mv = pd.read_csv(path_mv)
    df_mv = df_mv[['Stkcd', 'Trdmnt', 'Msmvttl']]
    df_mv.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date', 'Msmvttl': 'mv'}, inplace=True)
    df_mv['date'] = pd.to_datetime(df_mv['date'])
    # 单位是“千元”，转换为“元”
    df_mv['mv'] = df_mv['mv'] * 1000
    
    # --- 2.3 无风险利率 (Risk-Free Rate) ---
    print("加载无风险利率...")
    df_rf = pd.read_csv(path_rf)
    # 筛选月度、当期、1年期的存款利率作为无风险利率的代理是常见做法
    # 但您给的数据是回购利率，我们选择1个月期限(30d)作为短期利率代理
    df_rf = df_rf[(df_rf['Fresgn'] == 'M') & (df_rf['Datasgn'] == 'A') & (df_rf['Rptrm'] == '30d')]
    df_rf = df_rf[['Staper', 'Ezm0202']]
    df_rf.rename(columns={'Staper': 'date', 'Ezm0202': 'rf_annual_pct'}, inplace=True)
    df_rf['date'] = pd.to_datetime(df_rf['date'])
    # 将年化百分比利率转换为月度小数利率
    df_rf['rf'] = (1 + df_rf['rf_annual_pct'] / 100)**(1/12) - 1
    df_rf = df_rf[['date', 'rf']]
    
    # --- 2.4 账面价值 (Book Equity) ---
    print("加载账面价值数据...")
    df_be = pd.read_csv(path_be)
    # 筛选合并报表 (A类)，这是标准做法
    df_be = df_be[df_be['Typrep'] == 'A']
    df_be = df_be[['Stkcd', 'Accper', 'A003000000']]
    df_be.rename(columns={'Stkcd': 'stock_code', 'Accper': 'acc_date', 'A003000000': 'be'}, inplace=True)
    df_be.dropna(subset=['be'], inplace=True)
    # 确保账面价值 > 0
    df_be = df_be[df_be['be'] > 0]
    
    # Fama-French方法要求使用上一年末的财务数据，因此我们只保留年报数据
    df_be['acc_date'] = pd.to_datetime(df_be['acc_date'])
    df_be = df_be[df_be['acc_date'].dt.month == 12]
    df_be['year'] = df_be['acc_date'].dt.year
    df_be = df_be[['stock_code', 'year', 'be']]
    # 去重，以防一家公司在同一年份有多条年报记录
    df_be = df_be.drop_duplicates(subset=['stock_code', 'year'], keep='last')

    # 3. 合并数据

    # --- 3.1 合并回报率和市值 ---
    print("合并月度回报率和市值...")
    df = pd.merge(df_ret, df_mv, on=['stock_code', 'date'], how='inner')

    # --- 3.2 合并无风险利率 ---
    # 先将日期转为月初，方便合并
    df['date_monthly'] = df['date'].dt.to_period('M')
    df_rf['date_monthly'] = df_rf['date'].dt.to_period('M')
    print("合并无风险利率...")
    df = pd.merge(df, df_rf[['date_monthly', 'rf']], on='date_monthly', how='left')
    
    # --- 3.3 实现Fama-French逻辑，合并账面价值 ---
    # 关键步骤：使用t-1年末的BE，用于t年7月到t+1年6月的投资组合构建
    # 我们创建一个“财报年”的辅助列来进行匹配
    df['linking_year'] = np.where(df['date'].dt.month >= 7, df['date'].dt.year, df['date'].dt.year - 1)
    # 在BE数据中，使用财报的年份作为匹配键
    df_be.rename(columns={'year': 'linking_year'}, inplace=True)
    
    print("合并账面价值 (使用Fama-French时间逻辑)...")
    df = pd.merge(df, df_be, on=['stock_code', 'linking_year'], how='left')
    
    # 4. 计算最终变量并清理

    # --- 4.1 计算账面市值比 (B/M Ratio) ---
    # 我们需要使用每年6月末的市值来计算B/M，这个值在当年7月到次年6月保持不变
    # 因此，我们先获取每年6月的市值作为下一年的标准
    df_mv_june = df_mv[df_mv['date'].dt.month == 6]
    df_mv_june.rename(columns={'mv': 'mv_june'}, inplace=True)
    df_mv_june['linking_year'] = df_mv_june['date'].dt.year
    df = pd.merge(df, df_mv_june[['stock_code', 'linking_year', 'mv_june']], on=['stock_code', 'linking_year'], how='left')
    
    # 计算B/M. BE / ME_june
    df['bm_ratio'] = df['be'] / df['mv_june']
    
    # --- 4.2 计算超额收益率 ---
    df['ret_excess'] = df['ret'] - df['rf']
    
    # --- 4.3 清理和整理 ---
    print("最后清理和整理...")
    # 移除计算B/M后可能产生的无穷大值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 筛选掉核心数据缺失的行
    final_df = df.dropna(subset=['ret', 'mv', 'be', 'bm_ratio', 'ret_excess'])
    
    # 整理列的顺序
    final_df = final_df[[
        'stock_code', 'date', 'ret', 'rf', 'ret_excess', 'mv', 'be', 'bm_ratio'
    ]]

    # 5. 保存结果
    final_df.to_csv(output_path, index=False)
    print(f"数据整合完成！已保存至: {output_path}")
    print(f"最终数据集包含 {len(final_df)} 条记录。")
    print("数据集预览:")
    print(final_df.head())
    
    return final_df


if __name__ == '__main__':
    # 请将此路径修改为您本机的项目根目录
    # 脚本会从此路径下的'factor'子目录中读取数据
    project_base_path = r'/home/shimmer/projects/20250716_factor_investing/data'
    
    # 运行整合函数
    integrate_csmar_data(project_base_path)