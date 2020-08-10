import pysnowball as ball
import time,datetime
from prettytable import PrettyTable
import os


while True:
    code_list = [
        #{"name":"中国软件","code":"SH600536"},
        {"name":"创业50","code":"SZ159949"},
        {"name":"中国平安","code":"SH601318"},
        {"name":"中信证券","code":"SH600030"},
    ]
    for c in code_list:
        data = ball.quotec(c["code"])
        c["percent"] = str(data["data"][0]["percent"]) + '%'
        c["current"] = data["data"][0]["current"]
    os.system("cls")
    table = PrettyTable()
    for c in code_list:
        table.add_row(c.values())
    print(table)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"-"*20)
    time.sleep(3)

