# coding: utf-8
from sqlalchemy import Column, Float, String, DECIMAL, Integer,VARBINARY
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class WEB_ANALYSIS_API_ACCESS_NORMAL(Base):
    __tablename__ = 'web_analysis_api_access_normal'

    access_api = Column(VARBINARY(256),primary_key=True,nullable=False,comment="访问api")
    count = Column(Integer, comment='个数')


class WEB_ANALYSIS_API_ACCESS_UNNORMAL(Base):
    __tablename__ = 'web_analysis_api_access_unnormal'

    access_api = Column(VARBINARY(256),primary_key=True,nullable=False,comment="访问api")
    count = Column(Integer, comment='个数')


class WEB_ANALYSIS_IP_ACCESS_UNNORMAL(Base):
    __tablename__ = 'web_analysis_ip_access_unnormal'

    access_ip = Column(VARBINARY(256),primary_key=True,nullable=False,comment="访问ip")
    count = Column(Integer, comment='个数')
    country = Column(String(64),comment="国家")
    city = Column(String(64),nullable=False,comment="城市")
