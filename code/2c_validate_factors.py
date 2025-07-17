import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def validate_factors(our_factors_path, csm_factors_path, output_dir):
    """
    对比我们自己计算的因子和CSMAR提供的现成因子表。

    Args:
        our_factors_path (str): 我们自己计算的因子文件路径。
        csm_factors_path (str): CSMAR五因子文件路径。
        output_dir (str): 保存对比图表的目录。
    """
    print("### 开始进行因子交叉验证 ###")

    # 1. 加载数据
    print("加载我们自己计算的因子...")
    df_our = pd.read_csv(our_factors_path)
    df_our['date'] = pd.to_datetime(df_our['date'])
    df_our.rename(columns=lambda x: x + '_our' if x != 'date' else x, inplace=True)
    # 我们有UMD，但CSMAR表没有，先将其分离
    umd_our = df_our[['date', 'UMD_our']]
    df_our = df_our.drop(columns=['UMD_our'])


    print("加载CSMAR五因子表...")
    df_csm = pd.read_csv(csm_factors_path)
    
    # 2. 清洗和预处理CSMAR因子表
    # 定义我们知道是正确的列名
    csm_col_names = ['MarkettypeID', 'TradingMonth', 'Portfolios', 'RiskPremium2', 'SMB2', 'HML2', 'RMW2', 'CMA2']
    # 跳过前两行中英文表头，不使用文件中的任何行为列名(header=None)，而是直接使用我们定义的列名列表(names=...)
    df_csm = pd.read_csv(csm_factors_path, skiprows=2, header=None, names=csm_col_names)
    
    # 筛选：CSMAR表通常包含多种口径，我们需要选择一种
    # 这里我们做一个合理的假设：选择全A股市场(P9709)、投资组合类型1
    # 您可能需要根据数据说明书来确认这个选择
    market_type = 'P9709'
    portfolio_type = 1
    df_csm = df_csm[(df_csm['MarkettypeID'] == market_type) & (df_csm['Portfolios'] == portfolio_type)].copy()
    print(f"已筛选CSMAR数据: MarkettypeID='{market_type}', Portfolios={portfolio_type}")

    # 重命名和选择列
    df_csm = df_csm[['TradingMonth', 'SMB2', 'HML2', 'RMW2', 'CMA2']]
    df_csm.rename(columns={
        'TradingMonth': 'date',
        'SMB2': 'SMB_csm',
        'HML2': 'HML_csm',
        'RMW2': 'RMW_csm',
        'CMA2': 'CMA_csm'
    }, inplace=True)
    df_csm['date'] = pd.to_datetime(df_csm['date'])

    # 3. 合并两组因子数据
    # 使用inner join，确保只在两者都有数据的时间段内进行比较
    print("合并两组因子数据...")
    df_merged = pd.merge(df_our, df_csm, on='date', how='inner')

    if df_merged.empty:
        print("错误：两组因子数据没有重合的时间段，请检查日期格式或数据内容。")
        return

    print(f"数据合并完成，共有 {len(df_merged)} 个月的重合数据进行比较。")

    # 4. 进行统计比较
    # 4.1 计算相关系数矩阵
    correlation_matrix = df_merged[['SMB_our', 'SMB_csm', 'HML_our', 'HML_csm', 
                                     'RMW_our', 'RMW_csm', 'CMA_our', 'CMA_csm']].corr()
    
    print("\n--- 因子相关系数矩阵 ---")
    print("(我们关注的是 'our' 和 'csm' 对角线上的值，越高越好)")
    # 只看our和csm之间的相关性
    corr_pairs = {
        'SMB': correlation_matrix.loc['SMB_our', 'SMB_csm'],
        'HML': correlation_matrix.loc['HML_our', 'HML_csm'],
        'RMW': correlation_matrix.loc['RMW_our', 'RMW_csm'],
        'CMA': correlation_matrix.loc['CMA_our', 'CMA_csm'],
    }
    for factor, corr in corr_pairs.items():
        print(f"  - {factor} 因子相关性: {corr:.4f}")

    # 4.2 绘制对比图
    print("\n正在生成因子走势对比图...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    factors_to_plot = ['SMB', 'HML', 'RMW', 'CMA']
    for factor in factors_to_plot:
        plt.figure(figsize=(15, 6))
        plt.plot(df_merged['date'], df_merged[f'{factor}_our'], label=f'Our {factor}', alpha=0.8)
        plt.plot(df_merged['date'], df_merged[f'{factor}_csm'], label=f'CSMAR {factor}', alpha=0.8, linestyle='--')
        plt.title(f'Factor Comparison: {factor}')
        plt.xlabel('Date')
        plt.ylabel('Factor Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'comparison_{factor}.png'))
        plt.close()

    print(f"对比图已保存至: {output_dir}")
    print("\n### 交叉验证完成！ ###")


if __name__ == '__main__':
    # --- 请设置路径 ---
    # 我们自己计算的因子文件
    our_factors_file = '/home/shimmer/projects/20250716_factor_investing/data/factor/all_factors_monthly.csv'
    # 同事给的现成五因子文件
    csm_factors_file = '/home/shimmer/projects/20250716_factor_investing/data/factor/STK_MKT_FIVEFACMONTH.csv'
    # 保存对比图的文件夹
    output_plot_dir = '/home/shimmer/projects/20250716_factor_investing/data/factor/validation_plots'

    # 注意：我假设您已经将.xlsx文件另存为了.csv格式。如果仍是.xlsx，需要安装 'openpyxl' (`pip install openpyxl`)
    # 并在读取时使用 pd.read_excel()。为简化，建议转为CSV。

    validate_factors(our_factors_file, csm_factors_file, output_plot_dir)