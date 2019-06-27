import pandas as pd

df = pd.read_csv("y.csv")

nan_rows = df[df['dailyReturnReinv' or 'dailyReturnNoReinv'].isnull()]
#nan_rows.to_csv(r'nan_rows.csv')
tic_list=nan_rows.iloc[:,1]
tic_list=list(tic_list)
#print(tic_list)
new_df=df[~df['secID'].isin(tic_list)]
new_df.to_csv(r'new_y.csv')

stock = pd.read_csv("stock_data.csv")
new_stock = stock[~stock['secID'].isin(tic_list)]
new_stock.to_csv(r'new_stock.csv')

print(new_df)
