import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def draw():
    df = pd.read_csv('2018result.csv')
    df2 = pd.read_csv('new_real500.csv')
    invest_total = 10000000
    rank_interval = 100
    zhongzheng_inverval = 500
    dates = ['20180702','20180801','20180903','20181008','20181101','20181203']

    lists = list(df2['monthBeginDate'])
    new_list = []
    for i in lists:
        new_list.append(i.replace('-',''))
    df2['monthBeginDate'] = new_list
    df2['monthBeginDate'] = df2['monthBeginDate'].astype(np.int32)

    top = []
    real500 = []
    least = []

    top_2 = []
    real500_2 = []
    least_2 = []

    for day in dates:
        temp = (df.loc[df['date'] == (int(day) or str(day))])
        temp2 = (df2.loc[df2['monthBeginDate'] == (int(day) or str(day))])
        top100 = temp.nlargest(rank_interval, 'pred')
        last100 = temp.nsmallest(rank_interval, 'pred')

        top100_tickers = temp2.loc[temp2['ticker'].isin(top100['ticker'])]
        last100_tickers = temp2.loc[temp2['ticker'].isin(last100['ticker'])]

        real_profit = temp2['chgPct'].sum()
        pre_profit_top = top100_tickers['chgPct'].sum()
        pre_profit_last = last100_tickers['chgPct'].sum()

        # Discrete
        top.append(pre_profit_top*(invest_total/rank_interval)+invest_total)
        real500.append(real_profit*(invest_total/zhongzheng_inverval)+invest_total)
        least.append(pre_profit_last*(invest_total/rank_interval)+invest_total)

        # Accummulative
        if len(top_2) == 0:
            top_2.append(invest_total*(pre_profit_top/rank_interval+1))
            real500_2.append(invest_total*(real_profit/zhongzheng_inverval+1))
            least_2.append(invest_total*(pre_profit_last/rank_interval+1))
        else:
            top_2.append(top_2[-1]*(pre_profit_top/rank_interval+1))
            real500_2.append(real500_2[-1]*(real_profit/zhongzheng_inverval+1))
            least_2.append(least_2[-1]*(pre_profit_last/rank_interval+1))

    print(top)
    print(real500)
    print(least)
    print(top_2)
    print(real500_2)
    print(least_2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Back Test Based on LR with MSE = ')
    l1 = ax1.plot(dates, top, 'r')
    l2 = ax1.plot(dates, real500, 'b')
    l3 = ax1.plot(dates, least, 'g')
    plt.ylabel('Discrete')
    l4 = ax2.plot(dates, top_2, 'r')
    l5 = ax2.plot(dates, real500_2, 'b')
    l6 = ax2.plot(dates, least_2, 'g')

    # Create the legend
    fig.legend([l1, l2, l3],     # The line objects
               labels=['Top100', 'zhongzheng500', 'last100'],   # The labels for each line
               loc="center right",   # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               #title="Legend Title"  # Title for the legend
               )
    ax1.set_ylabel('discrete')
    ax2.set_ylabel('accumulative Total')
    plt.xlabel('time (month)')
    plt.show()

draw()
