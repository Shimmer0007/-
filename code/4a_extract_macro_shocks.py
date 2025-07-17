import pandas as pd
import numpy as np
import os
from statsmodels.tsa.api import VAR
from sklearn.linear_model import LinearRegression

def extract_macro_shocks(base_path):
    """
    加载所有宏观变量，使用VAR(1)模型提取意外冲击，
    并对市场超额收益进行正交化处理。
    """
    print("### 开始第四阶段-步骤A: 提取宏观意外冲击 ###")

    # 1. 定义文件路径
    path_macro_vars = os.path.join(base_path, 'all_macro_variables.csv')
    path_index = os.path.join(base_path, '指数文件/TRD_Index.csv')
    output_path = os.path.join(base_path, 'macro_shocks.csv')

    # 2. 加载数据
    print("加载所有宏观变量...")
    df_macro = pd.read_csv(path_macro_vars)
    df_macro['date'] = pd.to_datetime(df_macro['date'])
    df_macro.set_index('date', inplace=True)

    # 加载市场指数数据以计算市场超额收益
    print("加载市场指数收益率...")
    df_index = pd.read_csv(path_index)
    df_index.rename(columns={'Indexcd': 'index_code', 'Trddt': 'date', 'Retindex': 'ret'}, inplace=True)
    df_index['date'] = pd.to_datetime(df_index['date'])
    
    # 筛选沪深300指数，并计算月度收益率
    mkt_ret = df_index[df_index['index_code'] == 300].set_index('date')['ret']
    # 通过复合日收益得到月收益
    mkt_ret_monthly = (1 + mkt_ret).resample('ME').prod() - 1
    mkt_ret_monthly.name = 'mkt_ret'

    # 合并数据
    df = pd.merge(df_macro, mkt_ret_monthly, left_index=True, right_index=True, how='inner')
    df['mkt_excess_ret'] = df['mkt_ret'] - df['short_rate'] # 市场超额收益
    
    # 3. 拟合VAR(1)模型并提取残差（意外）
    print("拟合VAR(1)模型...")
    macro_cols = ['short_rate', 'term_spread', 'credit_spread', 'market_ep', 'market_vol', 'market_amihud']
    
    # 确保数据没有缺失值
    df_model_data = df[macro_cols].dropna()
    
    model = VAR(df_model_data)
    results = model.fit(1) # 1代表延迟阶数
    
    # 残差就是我们的“宏观意外”
    raw_shocks = results.resid
    raw_shocks.rename(columns=lambda x: x + '_shock', inplace=True)
    print("已从VAR模型中提取原始意外冲击。")

    # 4. 对市场超额收益进行正交化
    print("正在对意外冲击进行正交化处理...")
    df_to_ortho = pd.merge(raw_shocks, df[['mkt_excess_ret']], left_index=True, right_index=True, how='inner')
    
    ortho_shocks = pd.DataFrame(index=df_to_ortho.index)
    
    X = df_to_ortho[['mkt_excess_ret']]
    
    for shock_col in raw_shocks.columns:
        y = df_to_ortho[shock_col]
        
        # 线性回归: shock = beta * market_excess_return + residual
        reg = LinearRegression()
        reg.fit(X, y)
        
        # 预测值
        predictions = reg.predict(X)
        
        # 残差就是正交化后的冲击
        residuals = y - predictions
        ortho_shocks[shock_col + '_ortho'] = residuals
        
    print("正交化处理完成。")

    # 5. 保存结果
    ortho_shocks.reset_index(inplace=True)
    ortho_shocks.to_csv(output_path, index=False)
    
    print(f"\n### 宏观意外冲击已生成！###")
    print(f"最终结果已保存至: {output_path}")
    print("数据预览:")
    print(ortho_shocks.head())

    return ortho_shocks


if __name__ == '__main__':
    macro_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/mae'
    extract_macro_shocks(macro_base_path)