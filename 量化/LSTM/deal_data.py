from download_data import get_china_share,get_usa_share
import pandas as pd
import datetime,time

def get_pd_csv(code):
    #训练股票
    data  = get_china_share(code).set_index('date')
    #上证指数
    china_data_sh = get_china_share("sh.000001",quota="date,high,low,close").set_index('date')
    china_data_sh.rename(columns={'high':'sh_high', 'low':'sh_low', 'close':'sh_close'}, inplace = True)
    #深证指数
    china_data_sz = get_china_share("sz.399106",quota="date,high,low,close").set_index('date')
    china_data_sz.rename(columns={'high':'sz_high', 'low':'sz_low', 'close':'sz_close'}, inplace = True)
    #DJIA
    usa_data_djia  = get_usa_share("DJIA")[['Close','High','Low']]
    usa_data_djia.rename(index={'Date':'date'},columns={'Close':'djia_close', 'High':'djia_high', 'Low':'djia_low'}, inplace = True)
    #NDX
    usa_data_ndx  = get_usa_share("^NDX")[['Close','High','Low',]]
    usa_data_ndx.rename(index={'Date':'date'},columns={'Close':'ndx_close', 'High':'ndx_high', 'Low':'ndx_low'}, inplace = True)
    #to_csv避免每次跑批下载
    print("csv存储中.............")
    data.to_csv(code.replace(".","") + ".csv")
    china_data_sh.to_csv("SH.csv")
    china_data_sz.to_csv("SZ.csv")
    usa_data_djia.to_csv("DJIA.csv")
    usa_data_ndx.to_csv("NDX.csv")



def deal_usa_df(df):
    new_df = df.copy(deep=True)
    new_df = new_df.set_index("Date")
    last_day = df.iloc[0].Date
    for i in df.itertuples():
        i_day = i.Date
        delta_day = (datetime.datetime.strptime(i_day,'%Y-%m-%d') - datetime.datetime.strptime(last_day,'%Y-%m-%d')).days
        if delta_day <=1:last_day = i.Date
        else:
            for i in range(delta_day):
                new_day = datetime.datetime.strptime(last_day,'%Y-%m-%d') + datetime.timedelta(days = i+1)
                new_df.loc[new_day.strftime('%Y-%m-%d')] = list(new_df.loc[last_day])
                last_day = new_day.strftime('%Y-%m-%d')
        new_df.sort_index(inplace=True)
    return new_df


def get_csv_data(code):
    df = pd.read_csv(code.replace(".","") + ".csv").set_index('date')
    df = df.join(pd.read_csv("SH.csv",index_col=0))
    df = df.join(pd.read_csv("SZ.csv",index_col=0))
    df = df.join(deal_usa_df(pd.read_csv("NDX.csv")))
    df = df.join(deal_usa_df(pd.read_csv("DJIA.csv")))
    df.to_csv("train_data.csv")


if __name__ == '__main__':
    code = "sz.002223"
    get_pd_csv(code)
    get_csv_data(code)
