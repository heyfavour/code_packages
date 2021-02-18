#分析连续三种加仓   3日均线>5日均线>10日均线
#这两种情况下深股通的持股上涨的概率，来选择
#分析连续三日增长且增长超过0.4  5天一个周期 峰值超过60%
import pandas as pd
from db.session import session, engine
from sqlalchemy import distinct
from models.QT_models import HK_HOLD
import datetime
import matplotlib.pyplot as plt

def analysis(analysis_date, analysis_days=10):
    analysis_dates = session.query(distinct(HK_HOLD.hddate)).filter(HK_HOLD.hddate <= analysis_date
                                                                    ).order_by(HK_HOLD.hddate.desc()).all()[
                     0:analysis_days]
    analysis_dates = [i[0] for i in analysis_dates]

    query = session.query(HK_HOLD).filter(HK_HOLD.hddate.in_(analysis_dates)).order_by(HK_HOLD.hddate.asc())
    df = pd.read_sql(query.statement, query.session.bind, )
    for i, d in enumerate(analysis_dates):
        if i == 0:
            series_df = df[df.hddate == d].loc[:, ['hddate', 'scode', 'sharesrate']]
            series_df.rename(columns={"sharesrate": f"sharesrate_{i}"}, inplace=True)
        if i > 0:
            tmp_df = df[df.hddate == d].loc[:, ['scode', 'sharesrate']]
            tmp_df.rename(columns={"sharesrate": f"sharesrate_{i}"}, inplace=True)
            series_df = pd.merge(series_df, tmp_df, on='scode')
    # 1.连续3日加仓 数量较多
    constant_up = series_df[
        (series_df.sharesrate_0 > series_df.sharesrate_1) & (series_df.sharesrate_1 > series_df.sharesrate_2)&((series_df.sharesrate_0-series_df.sharesrate_2) > 1)]
    # 2.3日均线>5日均线>10日均线
    if constant_up.empty:return constant_up,constant_up
    series_df["daysavg10"] = series_df.apply(lambda row: get_row_avg(row, 10), axis=1)
    series_df["daysavg5"] = series_df.apply(lambda row: get_row_avg(row, 5), axis=1)
    series_df["daysavg3"] = series_df.apply(lambda row: get_row_avg(row, 3), axis=1)
    stable_up = series_df[(series_df.daysavg3 > series_df.daysavg5) & (series_df.daysavg5 > series_df.daysavg10)]
    return constant_up, stable_up


def compare_with_strategy(analysis_date, data=None):
    analysis_dates = session.query(distinct(HK_HOLD.hddate)).filter(HK_HOLD.hddate >= analysis_date
                                                                    ).order_by(HK_HOLD.hddate.asc()).all()
    analysis_dates = [v[0] for k, v in enumerate(analysis_dates) if k in (0, 1, 3, 5, 7)]
    query = session.query(HK_HOLD).filter(HK_HOLD.hddate.in_(analysis_dates)).order_by(HK_HOLD.hddate.asc())
    df = pd.read_sql(query.statement, query.session.bind, )
    for i, d in enumerate(analysis_dates):
        if i == 0:
            series_df = df[df.hddate == d].loc[:, ['hddate', 'scode', 'closeprice']]
            series_df.rename(columns={"closeprice": f"closeprice_{i}"}, inplace=True)
        if i > 0:
            tmp_df = df[df.hddate == d].loc[:, ['scode', 'closeprice']]
            tmp_df.rename(columns={"closeprice": f"closeprice_{i}"}, inplace=True)
            series_df = pd.merge(series_df, tmp_df, on='scode')
    df = pd.merge(data, series_df, on='scode')
    data = {}
    data['date'] = analysis_date
    print(df)
    data['percent_1'] = df[(df.closeprice_1 -  df.closeprice_0)/df.closeprice_0 > 0].scode.count() *100/ df.scode.count()
    data['percent_2'] = df[(df.closeprice_2 -  df.closeprice_0)/df.closeprice_0 > 0].scode.count() *100/ df.scode.count()
    data['percent_3'] = df[(df.closeprice_3 -  df.closeprice_0)/df.closeprice_0 > 0].scode.count() *100/ df.scode.count()
    data['percent_4'] = df[(df.closeprice_4 -  df.closeprice_0)/df.closeprice_0 > 0].scode.count() *100/ df.scode.count()
    return pd.DataFrame([data])

def get_row_avg(row, days):
    sum = 0
    for i in range(days):
        sum = sum + row[f"sharesrate_{i}"]
    return sum / days


def gen_plot(start_date,end_date):
    run_date = start_date = start_date
    end_date = end_date
    plot = None
    while run_date <= end_date:
        constant_up, stable_up = analysis(run_date, 10)
        if constant_up\
                .empty:
            run_date = int(
                (datetime.datetime.strptime(str(run_date), '%Y%m%d') + datetime.timedelta(days=1)).strftime("%Y%m%d"))
            continue
        constant = compare_with_strategy(run_date, constant_up)
        #stable = compare_with_strategy(run_date,stable_up)
        #df = pd.merge(constant, stable, on='date')
        df = constant
        if run_date == start_date:
            plot = df
        else:
            plot = plot.append(df,ignore_index=True)
        run_date = int((datetime.datetime.strptime(str(run_date), '%Y%m%d') + datetime.timedelta(days=1)).strftime("%Y%m%d"))
    plot.to_csv("plot.csv")

if __name__ == '__main__':
    gen_plot(20200803,20200825)
    df = pd.read_csv("plot.csv")
    df.plot(x='date',y=["percent_1"])
    df.plot(x='date',y=["percent_2"])
    df.plot(x='date',y=["percent_3"])
    df.plot(x='date',y=["percent_4"])
    plt.legend()
    plt.show()
