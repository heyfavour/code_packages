# coding: utf-8
from sqlalchemy import Column, Float, String, Text,DECIMAL,Integer
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class HK_HOLD(Base):
    __tablename__ = 'hk_hold'

    hddate = Column(Integer, primary_key=True, nullable=False,comment = '日期')
    hkcode = Column(String(16), primary_key=True, nullable=False,comment='HK代码')
    scode = Column(String(16),comment="A股代码")
    sname = Column(String(128),comment = 'A股名称')
    shareholdsum = Column(Float(asdecimal=True),comment = "持股量")
    sharesrate = Column(DECIMAL(10,4),comment = "持股比例")
    closeprice = Column(DECIMAL(10,4),comment="收盘价")
    zdf = Column(Float(asdecimal=True),doc="涨跌幅")
    shareholdprice = Column(Float(asdecimal=True),comment="持股市值")
    shareholdpriceone = Column(Float(asdecimal=True),comment="持股市值一日变化")
    shareholdpricefive = Column(Float(asdecimal=True),comment="持股市值五日变化")
    shareholdpriceten = Column(Float(asdecimal=True),comment="持股市值十日变化")
    market = Column(String(16),comment="市场")
    shareholdsumchg = Column(Float(asdecimal=True),comment="未知")
    zb = Column(Text,comment="未知")
    zzb = Column(Text,comment="未知")

    def columns(self):
        return [c.name for c in self.__table__.columns]


