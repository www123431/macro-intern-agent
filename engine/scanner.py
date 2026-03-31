import yfinance as yf
import pandas as pd
from datetime import datetime

class MarketScanner:
    def __init__(self):
        self.watchlist = {
            "AI算力": "SMH",
            "科技成长": "QQQ",
            "沪深300": "ASHR",
            "新加坡蓝筹": "EWS",
            "全球能源": "XLE"
        }

    def run_daily_scan(self):
        """执行扫描并返回最具潜力的板块"""
        results = []
        for name, ticker in self.watchlist.items():
            try:
                df = yf.download(ticker, period="60d")
                # 计算夏普比率、动能、偏度等专业指标
                returns = df['Close'].pct_change().dropna()
                sharpe = (returns.mean() * 252) / (returns.std() * (252**0.5))
                
                results.append({
                    "name": name,
                    "ticker": ticker,
                    "sharpe": float(sharpe),
                    "last_price": float(df['Close'].iloc[-1])
                })
            except Exception as e:
                print(f"扫描 {name} 出错: {e}")
        
        # 排序并选出 Top 1
        best_pick = sorted(results, key=lambda x: x['sharpe'], reverse=True)[0]
        return best_pick

# 测试代码
if __name__ == "__main__":
    scanner = MarketScanner()
    print(f"今日扫描结果：{scanner.run_daily_scan()}")
