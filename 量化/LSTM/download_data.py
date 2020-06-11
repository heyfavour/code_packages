#国内股票
import baostock
import pandas as pd
import datetime
#国外股票
import yfinance as yf

def get_china_share(code,quota="date,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",start_date=19970101,end_date=20301231):
    try:
        login = baostock.login()
        if login.error_code != "0":raise Exception(login.error_msg)
        rs = baostock.query_history_k_data_plus(
            code,quota,
            start_date = datetime.datetime.strptime(str(start_date), '%Y%m%d').strftime("%Y-%m-%d"),
            end_date = datetime.datetime.strptime(str(end_date), '%Y%m%d').strftime("%Y-%m-%d"),
            frequency = "d",
            adjustflag = "2",#复权状态(1：后复权， 2：前复权，3：不复权）
        )
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        return result
    except Exception as e:
        print(str(e))
    finally:
        baostock.logout()

def get_usa_share(code,start=19970101, end=20301231):
    start =  datetime.datetime.strptime(str(start), '%Y%m%d').strftime("%Y-%m-%d")
    end =  datetime.datetime.strptime(str(end), '%Y%m%d').strftime("%Y-%m-%d")
    ticker = yf.Ticker(code).history(start=start,end=end,interval='1d')
    return ticker

if __name__ == '__main__':
    #code = "sz.002223"
    #data = get_china_share(code)
    code = "^NDX"#深SZ  沪 SS 但本下载方式较慢 且数据不够全 例如"002223.SZ"
    data = get_usa_share(code)
    data.to_csv(code + ".csv")