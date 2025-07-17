import pandas as pd
import os

def process_rates_data(base_path):
    """
    【最终修正版】处理宏观指标1和2：短期利率 和 期限利差。
    修正了国债收益率期限的筛选条件。
    """
    print("### 开始处理宏观指标1 & 2: 短期利率与期限利差 ###")

    # 1. 定义文件路径
    path_shibor = os.path.join(base_path, '短期利率/FE_SHIBOR.csv')
    path_treasury = os.path.join(base_path, '中债国债收益率/BND_TreasYield.csv')
    output_path = os.path.join(base_path, 'macro_variables_part1.csv')

    # --- 处理指标1: 短期利率 (Short Rate) ---
    print("正在处理SHIBOR数据以获取短期利率...")
    df_shibor = pd.read_csv(path_shibor)
    df_shibor.rename(columns={'TradingDate': 'date', 'Term': 'term', 'IntersetRate': 'rate'}, inplace=True)
    df_shibor['date'] = pd.to_datetime(df_shibor['date'])
    if 'Currency' in df_shibor.columns:
        df_shibor = df_shibor[df_shibor['Currency'] == 'CNY']
    df_shibor = df_shibor[df_shibor['term'] == '7天']
    df_shibor.set_index('date', inplace=True)
    monthly_shibor = df_shibor['rate'].resample('ME').last()
    short_rate = monthly_shibor / 100
    short_rate.name = 'short_rate'
    print("短期利率处理完成。")


    # --- 处理指标2: 期限利差 (Term Spread) ---
    print("正在处理国债收益率数据以获取期限利差...")
    df_treasury = pd.read_csv(path_treasury)
    df_treasury.rename(columns={'Trddt': 'date', 'Cvtype': 'cv_type', 'Yeartomatu': 'term', 'Yield': 'yield'}, inplace=True)
    df_treasury['date'] = pd.to_datetime(df_treasury['date'])
    df_treasury = df_treasury[df_treasury['cv_type'] == 1]
    
    # 【修正】使用数字 1.0 和 10.0 进行筛选，而不是字符串 '1y' 和 '10y'
    df_1y = df_treasury[df_treasury['term'] == 1.0].set_index('date')['yield']
    df_10y = df_treasury[df_treasury['term'] == 10.0].set_index('date')['yield']

    monthly_1y = df_1y.resample('ME').last()
    monthly_10y = df_10y.resample('ME').last()
    term_spread = (monthly_10y - monthly_1y) / 100
    term_spread.name = 'term_spread'
    print("期限利差处理完成。")

    # --- 合并结果 ---
    print("合并短期利率和期限利差...")
    df_final = pd.concat([short_rate, term_spread], axis=1)
    df_final.dropna(inplace=True)
    df_final.reset_index(inplace=True)
    
    # --- 保存结果 ---
    df_final.to_csv(output_path, index=False)
    print(f"处理完成！指标1和2已保存至: {output_path}")
    print("数据预览:")
    print(df_final.head())

    return df_final


if __name__ == '__main__':
    macro_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/mae'
    process_rates_data(macro_base_path)