#! /usr/bin/python
# Created by Kirk Hayes (l0gan)
# Part of myBFF, SMB guessing
from ftplib import FTP
from core.nonHTTPModule import nonHTTPModule
import re


class FTPbrute(nonHTTPModule):
    def __init__(self, config, display, lock):
        super(FTPbrute, self).__init__(config, display, lock)
        self.fingerprint="FTP"
        self.response=""
        self.protocol="ftp"

    def somethingCool(self, config, userID, password, server_name, conn, connection):
            print("[+]      Listing contents of the root directory...")
            conn.dir()

    def connectTest(self, config, userID, password, server_name, proxy):
        port = '21'
        if ":" in server_name:
            port = re.sub("^.*:", '', server_name)
            server_name = re.sub(":.*$", '', server_name)
        try:
            conn = FTP()
            conne = conn.connect(server_name, port)
            connection = conn.login(userID, password)
            if "230" in connection:
                print("[+]  User Credentials Successful: " + config["USERNAME"] + ":" + config["PASSWORD"])
                self.somethingCool(config, userID, password, server_name, conn, connection)
        except Exception as e:
            print("[-]  Login Failed for: " + config["USERNAME"] + ":" + config["PASSWORD"])
