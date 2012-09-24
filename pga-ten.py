#!/usr/bin/env python

import time

import comedi_interface
from comedi_interface import CHANNEL_A0 as A0
from comedi_interface import CHANNEL_A1 as A1
from comedi_interface import CHANNEL_RELAY as RELAY

def test(comedi):
    i = 0
    while True:
        comedi.send(A0, i & 1)
        comedi.send(A1, (i & 2) >> 1)
        comedi.send(RELAY, (i&4) >> 2)
        time.sleep(1)
        i += 1

def amp(comedi, a0, a1):
    comedi.send(A0, a0)
    comedi.send(A1, a1)


def main():
    comedi = comedi_interface.Comedi()
    comedi.send(RELAY, 0)
    amp(comedi, 1, 0)

    #test(comedi)

main()

