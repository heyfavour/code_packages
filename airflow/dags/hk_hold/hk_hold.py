import akshare as ak
from .models.QT_models import HK_HOLD
from .db.session import session,engine
import pandas as pd

def delete_table(runday):
    session.query(HK_HOLD).filter_by(HDDATE=int(runday)).delete()

def hk_hold_into_db(runday):
    delete_table(runday)
    df = ak.stock_em_hsgt_stock_statistics(market="北向持股", start_date=str(runday), end_date=str(runday))
    if df.empty:return
    df['HDDATE'] = df['HDDATE'].apply(lambda x:int(x[0:10].replace("-","")))
    df['CLOSEPRICE'] = df['CLOSEPRICE'].apply(lambda x:0 if x=='-' else x)
    df['ZDF'] = df['ZDF'].apply(lambda x:0 if x=='-' else x)
    df ['Zb'] = df ['Zb'].apply(pd.to_numeric,errors ='raise')
    df ['Zzb'] = df ['Zzb'].apply(pd.to_numeric,errors ='raise')
    df.to_sql("hk_hold", engine,if_exists="append", index=False)
