#!/usr/bin/env python
# -*- coding:utf-8 -*-_

import sys
import platform

def md5_count(yingjian_info):
    import hashlib
    import random
    random.seed(4)

    demo_val = yingjian_info
    md5_val = hashlib.md5(demo_val.encode('utf-8')).hexdigest()
    md5_val_list = list(md5_val)
    random.shuffle(md5_val_list)
    md5_val = ''.join(md5_val_list)
    return md5_val

def jiami1(yingjian_info):
    return md5_count(yingjian_info)

'''
获取硬件信息
'''
sys_name = platform.system()

if sys_name == "Windows":
    import wmi, uuid

import os


def getwindowsboardinfo():
    print("\n-------------主板序列号-------------")
    boardall = wmi.WMI().Win32_BaseBoard()

    res = ""
    for disk in boardall:
        res += disk.SerialNumber.strip()
    print(res)

    return res


def getwindowsdiskinfo():
    print("\n-------------硬盘序列号-------------")
    diskall = wmi.WMI().Win32_DiskDrive()

    res = ""
    for disk in diskall:
        res += disk.SerialNumber.strip()
    print(res)

    return res


def getwindowsmacinfo():
    print("\n-------------mac地址-------------")
    macall = wmi.WMI().Win32_NetworkAdapterConfiguration(IPEnabled=1)

    res = ""
    for mac in macall:
        res += mac.MACAddress
    print(res)
    # node = uuid.getnode()
    # print(node, "@@@@@")
    # res = uuid.UUID(int = node).hex[-12:]

    return res


def getwindowscpuinfo():
    print("\n------------CPU序列号-------------")
    cpuall = wmi.WMI().Win32_Processor()

    res = ""
    for cpu in cpuall:
        res += cpu.ProcessorId.strip()
    print(res)

    return res


def getwindowsinfo():
    boardstr = getwindowsboardinfo()
    diskstr = getwindowsdiskinfo()
    macstr = getwindowsmacinfo()
    cpustr = getwindowscpuinfo()

    return "".join([boardstr, diskstr, macstr, cpustr])


def getlinuxboardinfo():
    print("\n-------------主板序列号-------------")
    p = os.popen("dmidecode -t baseboard | grep \"Serial Number\"")

    res = ""
    for line in p.read().split("\n"):
        line = line.strip()
        if not line: continue
        sn = line.split(":")[-1].strip()
        # res += (sn if sn != "Not Specified" else "")
        res += sn

    print(res)

    return res


def getlinuxdiskinfo():
    print("\n-------------硬盘序列号-------------")
    p = os.popen("lsblk --nodeps -no serial /dev/sda")
    res = p.read().replace("\n", "")
    print(res)

    return res


def getlinuxmacinfo():
    print("\n-------------mac地址-------------")
    res = ""

    for line in os.popen("/sbin/ifconfig"):
        if "Ether" in line:
            res = line.split()[1]
            break
    print(res)
    return res


def getlinuxcpuinfo():
    print("\n-------------CPU序列号-------------")
    p = os.popen("dmidecode -t processor | grep \"Serial Number\"")

    # print(p, p.read(), p.readline(), '@@@@@@@@@@@')
    # print(p.read(), '@@@@@@@@@')
    res = ""
    for line in p.read().split("\n"):
        line = line.strip()
        if not line: continue
        sn = line.split(":")[-1].strip()
        res += sn
    # res += (sn if sn != "Not Specified" else "")

    print(res)
    return res


def getlinuxinfo():
    boardstr = getlinuxboardinfo()
    diskstr = getlinuxdiskinfo()
    macstr = getlinuxmacinfo()
    cpustr = getlinuxcpuinfo()

    return "".join([boardstr, diskstr, macstr, cpustr])


def work():
    infostr = ""
    if sys_name == "Windows":
        infostr = getwindowsinfo()
    elif sys_name == "Linux":
        infostr = getlinuxinfo()
    else:
        pass

    print("\n-------------结果串为-------------")
    print(infostr)

    return infostr


if __name__ == '__main__':
    yingjian_info = 'WJG13XX0'
    # yingjian_info = 'VM18BS013556WJG13XX002:42:aa:98:67:b0Not SpecifiedNot Specified'+'abcdfghk'
    yingjian_info = work()+'abcdfghk'
    jiami1_result = jiami1(yingjian_info)
    print(jiami1_result)
    with open('key1','w') as fw:
        fw.write(jiami1_result)
