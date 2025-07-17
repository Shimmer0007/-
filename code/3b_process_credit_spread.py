import pandas as pd
import os

def process_credit_spread(base_path):
    """
    【最终修正版】处理宏观指标3：信用利差。
    修正了最后合并时的日期对齐问题。
    """
    print("### 开始处理宏观指标3: 信用利差 ###")

    # 1. 定义文件路径
    path_rating = os.path.join(base_path, '债券及主体评级情况表/BND_Rating.csv')
    path_corp_yield1 = os.path.join(base_path, '债券月交易信息表/BND_Bndmt.csv')
    path_corp_yield2 = os.path.join(base_path, '债券月交易信息表/BND_Bndmt1.csv')
    path_treasury = os.path.join(base_path, '中债国债收益率/BND_TreasYield.csv')
    path_part1_vars = os.path.join(base_path, 'macro_variables_part1.csv')
    output_path = os.path.join(base_path, 'macro_variables_part2.csv')

    # --- 步骤A: 计算月度AAA级企业债平均收益率 ---
    print("正在处理企业债数据...")
    df_corp1 = pd.read_csv(path_corp_yield1)
    df_corp2 = pd.read_csv(path_corp_yield2)
    df_corp_yield = pd.concat([df_corp1, df_corp2], ignore_index=True)
    df_corp_yield.rename(columns={'Liscd': 'bond_code', 'Trdmnt': 'date_str', 'Clsyield': 'yield'}, inplace=True)
    df_corp_yield['date'] = pd.to_datetime(df_corp_yield['date_str'], format='%Y-%m')
    df_corp_yield = df_corp_yield[['bond_code', 'date', 'yield']].dropna()

    print("正在处理债券评级数据...")
    df_rating = pd.read_csv(path_rating)
    df_rating.rename(columns={'Liscd': 'bond_code', 'Ctcr': 'rating'}, inplace=True)
    aaa_bonds = df_rating[df_rating['rating'] == 'AAA']['bond_code'].unique()
    df_corp_yield_aaa = df_corp_yield[df_corp_yield['bond_code'].isin(aaa_bonds)]
    avg_aaa_yield = df_corp_yield_aaa.groupby('date')['yield'].mean()
    avg_aaa_yield.name = 'avg_aaa_yield'
    print("AAA级企业债月度平均收益率计算完成。")

    # --- 步骤B: 获取月度国债收益率基准 ---
    print("正在处理国债收益率数据作为基准...")
    df_treasury = pd.read_csv(path_treasury)
    df_treasury.rename(columns={'Trddt': 'date', 'Cvtype': 'cv_type', 'Yeartomatu': 'term', 'Yield': 'yield'}, inplace=True)
    df_treasury['date'] = pd.to_datetime(df_treasury['date'])
    df_treasury = df_treasury[df_treasury['cv_type'] == 1]
    df_10y = df_treasury[df_treasury['term'] == 10.0].set_index('date')['yield']
    # 注意：这里我们先不把日期转成月初，直接用月末的日期作为索引
    gov_yield_10y = df_10y.resample('ME').last()
    gov_yield_10y.name = 'gov_yield_10y'
    print("国债基准收益率处理完成。")
    
    # --- 步骤C: 计算信用利差并合并所有数据 ---
    print("正在计算信用利差并合并所有指标...")
    # 【修正】在合并AAA收益率和国债收益率之前，先统一日期标准
    # 将avg_aaa_yield的月初日期索引，转换为月末日期，以匹配国债收益率的索引
    avg_aaa_yield.index = avg_aaa_yield.index + pd.offsets.MonthEnd(0)
    
    df_spread_calc = pd.concat([avg_aaa_yield, gov_yield_10y], axis=1).dropna()
    df_spread_calc['credit_spread'] = (df_spread_calc['avg_aaa_yield'] - df_spread_calc['gov_yield_10y']) / 100
    
    # 加载part1的数据
    df_part1 = pd.read_csv(path_part1_vars)
    df_part1['date'] = pd.to_datetime(df_part1['date'])
    
    # 现在df_spread_calc的索引是月末日期，可以和df_part1的'date'列正确合并了
    df_final = pd.merge(df_part1, df_spread_calc[['credit_spread']], on='date', how='left')
    
    df_final.to_csv(output_path, index=False)
    print(f"处理完成！指标1,2,3已合并保存至: {output_path}")
    print("数据预览:")
    print(df_final.head())

    return df_final


if __name__ == '__main__':
    macro_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/mae'
    process_credit_spread(macro_base_path)