import pandas as pd
import numpy as np
import yfinance as yf
from backtesting.test import SMA, EMA, RSI, CD, Value, Z_Score
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, cross, plot_heatmaps
from datetime import datetime

symbol = '069500.KS'
#ticker_KR = '^KS11'
ticker_US = '^GSPC'
#ticker_DXY = 'DX=F'
today = datetime.today().strftime('%Y-%m-%d')
start, end = '2021-12-31', today
#ohlcv_KR = yf.download(tickers=ticker_KR, start=start, end=today)
ohlcv_US = yf.download(tickers=ticker_US, start=start, end=today)
#ohlcv_DXY = yf.download(tickers=ticker_DXY, start=start, end=today)
vix = yf.download(tickers='^VIX', start=start, end=today).Close

class RSICross(Strategy):
    rsi_n, rsi_m = 14, 9
    rsi_under, rsi_upper = 30, 70
    vix_under, vix_upper = 20, 34
    fast, slow, smooth = 12, 26, 9
    window = 60

    def init(self):
        price = self.data.Close
        # =======================MA=======================
        ma = EMA  # the type of MA; EMA
        self.ma3 = self.I(ma, price, 3, plot=True, color='red')
        self.ma5 = self.I(ma, price, 5, plot=True, color='orange')
        self.ma20 = self.I(ma, price, 20, plot=True, color='yellow')
        self.ma30 = self.I(ma, price, 30, plot=True, color='green')
        self.ma60 = self.I(ma, price, 60, plot=True, color='blue')
        self.ma120 = self.I(ma, price, 120, plot=True, color='purple')

        # =====================RSI=========================
        self.rsi = self.I(RSI, price, self.rsi_n, plot=True, overlay=False, name='RSI')
        self.rsi_z = self.I(Z_Score, self.rsi, 14, plot=True, overlay=False, name='RSI 2 weeks Z-score')
        self.vix = self.I(Value, vix, plot=True, overlay=False, name='VIX')
        self.vix_z = self.I(Z_Score, self.vix, 14, plot=True, overlay=False, name='VIX 2 weeks Z-score')
        self.vix_z2 = self.I(Z_Score, self.vix, 28, plot=True, overlay=False, name='VIX 4 weeks Z-Score')
        self.vix_z3 = self.I(Z_Score, self.vix, 90, plot=False, overlay=False, name='VIX 3 months Z-Score')

        # =====================MACD=========================
        plot_macd = False
        self.macd_3_5 = self.I(CD, self.ma3, self.ma5, plot=plot_macd, name='ma 3-5')
        self.macd_3_5_Z = self.I(Z_Score, self.macd_3_5, plot=plot_macd, overlay=False, name='ma 3-5(Z)')
        self.macd_20_60 = self.I(CD, self.ma20, self.ma60, plot=plot_macd, name='ma 20-60')
        self.macd_20_60_Z = self.I(Z_Score, self.macd_20_60, plot=plot_macd, overlay=False, name='ma 20-60(Z)')

    def next(self):
        # 1. rsi가 rsi 하한(30)을 하향돌파 하는 경우 -> 과매도 가능성
        if crossover(30, self.rsi):
            # 1-1. VIX가 30미만인 경우 -> 홀드
            if self.vix<30:
                # hold
                pass
            # 1-2. VIX가 30이상인 경우 -> 매수
            elif self.vix>=30:
                # 1-2-1. short position인 경우
                if self.position.is_short:
                    self.position.close()
                    self.buy(size=0.5)
                # 1-2-2. hold/long position인 경우 -> buy
                else:
                    self.buy()
        # 2. rsi가 rsi 상한(70)을 상향돌파 하는 경우 -> 과매수 가능성
        elif crossover(self.rsi, 70):
            # 2-1. long position인 경우
            if self.position.is_long:
                # position close
                self.position.close()
                if self.vix < self.vix_under:
                    self.sell()
                else:
                    pass
            # 2-2. short position인 경우
            elif self.position.is_short:
                # position hold
                pass
            # 2-3. position이 없는 경우
            else:
                # 매도
                self.sell()
        elif crossover(self.vix_z2, 6):
            self.buy()

        # VIX overshooting -> 매수
        elif crossover(self.vix, self.vix_upper):
            if self.position.is_short:
                self.position.close()
            elif self.position.is_long:
                self.buy(size=0.5)
            else:
                self.buy(size=0.8)
        # VIX Undershooting -> 매도
        """elif crossover(20, self.vix):
            if self.position.is_short:
                self.sell(size=0.5)
            elif self.position.is_long:
                self.position.close()
            else:
                self.sell(size=0.5)"""


bt = Backtest(ohlcv_US, RSICross, commission=0.025, trade_on_close=True)
stats = bt.run()
stats, bt.plot(plot_volume=False, plot_return=False)

"stats, heatmap = bt.optimize(vix_under=range(17, 22, 1),
                             vix_upper=range(33, 36, 1),
                             maximize='Equity Final [$]',
                             return_heatmap=True)
"

# Backtest.optimize()