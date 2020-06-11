#!/usr/bin/python
# -*- coding: UTF-8 -*-

import socket
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(('127.0.0.1',8079))
server.listen(5)

while True:
    client,client_addr  = server.accept()
    client.send("请问您有什么问题么".encode("utf-8"))
    while True:
        client_send  = client.recv(1024).decode("utf-8")
        print(client_send)
        server_send = input('回复:{client_addr}'.format(client_addr=client_addr)).encode("utf-8")
        client.send(server_send)