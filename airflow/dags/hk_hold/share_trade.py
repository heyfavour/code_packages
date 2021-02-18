import akshare as ak
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__)))
from models.QT_models import SHARE_TRADE_DATE
from db.session import session, engine


def delete_table():
    session.query(SHARE_TRADE_DATE).delete()


def trade_date_into_db():
    delete_table()
    df = ak.tool_trade_date_hist_sina()
    df['trade_date'] = df['trade_date'].apply(lambda x: int(x[0:10].replace("-", "")))
    df.to_sql("share_trade_date", engine, if_exists="append", index=False)
    return "success"

if __name__ == '__main__':
    trade_date_into_db()
