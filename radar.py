import yfinance as yf
import pandas as pd
import numpy as np

class AutonomousRadar:
    def __init__(self):
        # 资产池：涵盖 A 股、美股、新加坡市场及基建
        self.pool = {
            "沪深300": "ASHR",
            "AI算力": "SMH",
            "新加坡蓝筹": "EWS",
            "科技成长": "QQQ",
            "数据中心REITs": "SRRE" # 举例：与AI相关的基建
        }

    def scan_market_pulse(self):
        """扫描全池，寻找‘异常稳健’的资产"""
        performance = []
        for name, ticker in self.pool.items():
            try:
                # 获取 60 天数据
                df = yf.download(ticker, period="60d", progress=False)
                returns = df['Close'].pct_change().dropna()
                
                # 计算核心指标
                vol = returns.std() * np.sqrt(252)
                ann_return = returns.mean() * 252
                sharpe = ann_return / vol if vol > 0 else 0
                
                performance.append({
                    "name": name,
                    "sharpe": sharpe,
                    "volatility": vol,
                    "momentum": (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
                })
            except:
                continue
        
        # 排序：取夏普比率最高的资产
        top_pick = sorted(performance, key=lambda x: x['sharpe'], reverse=True)[0]
        return top_pick
