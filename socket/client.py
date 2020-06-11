#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名：client.py

import socket  # 导入 socket 模块

client = socket.socket()  # 创建 socket 对象
client.connect(('127.0.0.1', 8079))
while True:
    server_send = client.recv(1024).decode("utf-8")
    print("SERVER_SEND:",server_send)
    client_send = input("CLIENT_SEND:")
    client.send(client_send.encode("utf-8"))
