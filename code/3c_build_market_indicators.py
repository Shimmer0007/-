import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def load_all_stock_data(path_pattern):
    """
    加载所有匹配路径模式的个股日交易数据CSV文件并合并。
    """
    all_files = glob.glob(path_pattern)
    if not all_files:
        raise FileNotFoundError(f"在指定的路径模式下没有找到任何文件: {path_pattern}")
    
    print(f"找到 {len(all_files)} 个个股数据文件，开始合并...")
    df_list = []
    # 使用tqdm显示加载进度
    for filename in tqdm(all_files, desc="正在加载个股数据文件"):
        # 增加error_bad_lines=False和warn_bad_lines=True来处理潜在的CSV格式问题
        df_list.append(pd.read_csv(filename, on_bad_lines='skip'))
    
    df_combined = pd.concat(df_list, axis=0, ignore_index=True)
    print("所有个股数据合并完成。")
    return df_combined

def build_market_indicators(base_path):
    """
    【最终修正版】构建指标4,5,6并与之前的指标合并。
    修正了指标4（市场估值）的计算逻辑。
    """
    print("### 开始构建剩余的宏观指标 (估值、波动率、流动性) ###")

    # 1. 定义文件路径
    stock_data_dir = os.path.join(base_path, '个股日交易衍生指标')
    stock_data_pattern = os.path.join(stock_data_dir, '*.csv')
    path_index = os.path.join(base_path, '指数文件/TRD_Index.csv')
    path_part2_vars = os.path.join(base_path, 'macro_variables_part2.csv')
    output_path = os.path.join(base_path, 'all_macro_variables.csv')

    # 2. 加载并预处理数据
    df_stock_daily = load_all_stock_data(stock_data_pattern)
    df_stock_daily.rename(columns={
        'TradingDate': 'date', 'Symbol': 'stock_code', 'PE': 'pe',
        'CirculatedMarketValue': 'mv', 'ChangeRatio': 'ret', 'Amount': 'amount'
    }, inplace=True)
    df_stock_daily['date'] = pd.to_datetime(df_stock_daily['date'])
    # 确保关键列是数值型，无法转换的设为NaN
    numeric_cols = ['pe', 'mv', 'ret', 'amount']
    for col in numeric_cols:
        df_stock_daily[col] = pd.to_numeric(df_stock_daily[col], errors='coerce')

    df_index = pd.read_csv(path_index)
    df_index.rename(columns={'Indexcd': 'index_code', 'Trddt': 'date', 'Retindex': 'ret'}, inplace=True)
    df_index['date'] = pd.to_datetime(df_index['date'])

    # --- 3. 构建指标 ---

    # 指标4: 市场估值 (Market E/P)
    print("\n>>> 正在构建指标4: 市场整体估值 (E/P)...")
    pe_data = df_stock_daily[['date', 'stock_code', 'pe', 'mv']].copy()
    pe_data = pe_data[pe_data['pe'] > 0].dropna()
    
    # 【修正】定义一个函数，用于计算每个月截面的加权平均PE
    def get_weighted_pe(group):
        # group是某一个月的所有数据
        if group.empty or group['mv'].sum() == 0:
            return np.nan
        # 使用np.average直接计算加权平均，更高效
        return np.average(group['pe'], weights=group['mv'])

    # 按月分组，并对每个月的组应用上面定义的函数
    weighted_pe_series = pe_data.set_index('date').groupby(pd.Grouper(freq='ME')).apply(get_weighted_pe)
    market_ep = 1 / weighted_pe_series
    market_ep.name = 'market_ep'
    print("市场估值(E/P)处理完成。")

    # 指标5: 市场波动率 (Volatility)
    print("\n>>> 正在构建指标5: 市场波动率...")
    vol_data = df_index[df_index['index_code'] == 300][['date', 'ret']].copy()
    vol_data.set_index('date', inplace=True)
    market_vol = vol_data['ret'].resample('ME').std() * np.sqrt(252)
    market_vol.name = 'market_vol'
    print("市场波动率处理完成。")
    
    # 指标6: 市场流动性 (Amihud)
    print("\n>>> 正在构建指标6: 市场流动性 (Amihud)...")
    liq_data = df_stock_daily[['date', 'stock_code', 'ret', 'amount', 'mv']].copy().dropna()
    liq_data = liq_data[liq_data['amount'] > 0]
    liq_data['amihud'] = abs(liq_data['ret']) * 1000000 / liq_data['amount'] # 放大因子以避免数值过小
    
    def get_weighted_amihud(group):
        if group.empty or group['mv'].sum() == 0:
            return np.nan
        return np.average(group['amihud'], weights=group['mv'])
        
    market_amihud = liq_data.set_index('date').groupby(pd.Grouper(freq='ME')).apply(get_weighted_amihud)
    market_amihud.name = 'market_amihud'
    print("市场流动性(Amihud)处理完成。")

    # --- 4. 最终合并 ---
    print("\n>>> 正在合并所有宏观指标...")
    df_part2 = pd.read_csv(path_part2_vars)
    df_part2['date'] = pd.to_datetime(df_part2['date'])
    df_part2.set_index('date', inplace=True)
    
    df_new_vars = pd.concat([market_ep, market_vol, market_amihud], axis=1)
    
    df_final = pd.merge(df_part2, df_new_vars, left_index=True, right_index=True, how='left')
    df_final.dropna(inplace=True)
    df_final.reset_index(inplace=True)

    # --- 保存结果 ---
    df_final.to_csv(output_path, index=False)
    print(f"\n### 数据准备收官！###")
    print(f"所有6个宏观指标已合并保存至: {output_path}")
    print("最终数据预览:")
    print(df_final.head())

    return df_final


if __name__ == '__main__':
    macro_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/mae'
    build_market_indicators(macro_base_path)