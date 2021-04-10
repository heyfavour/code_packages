import akshare as ak
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__)))
from models.QT_models import HK_HOLD,SHARE_TRADE_DATE
from db.session import session, engine


def delete_table(runday):
    session.query(HK_HOLD).filter_by(date_id=int(runday)).delete()


def hk_hold_into_db(runday):
    print(f"hk_hold_into_db=>{runday}")
    delete_table(runday)
    #check_date
    trade_date = session.query(SHARE_TRADE_DATE).filter_by(trade_date=int(runday)).first()
    if not trade_date:return "today no trade"
    try:
        df = ak.stock_em_hsgt_stock_statistics(symbol="北向持股", start_date=str(runday), end_date=str(runday))
    except ValueError:
        return "today no trade"
    columns = {
        '持股日期': 'date_id',
        '股票代码': 'code',
        '股票简称': 'name',
        '当日收盘价': 'price',
        '当日涨跌幅': 'amplitude',
        '持股数量': 'numbers',
        '持股市值': 'market_value',
        '持股数量占发行股百分比': 'share_percent',
        '持股市值变化-1日': 'change_one_days',
        '持股市值变化-5日': 'change_five_days',
        '持股市值变化-10日': 'change_ten_days',
    }
    df.rename(columns=columns, inplace=True)
    df['date_id'] = df['date_id'].apply(lambda x: int(x[0:10].replace("-", "")))
    df['price'] = df['price'].apply(lambda x: 0 if x == '-' else x)
    df['amplitude'] = df['amplitude'].apply(lambda x: 0 if x == '-' else x)
    df.to_sql("hk_hold", engine, if_exists="append", index=False)
    #print(df)
    return "success"

if __name__ == '__main__':
    hk_hold_into_db('20210402')

