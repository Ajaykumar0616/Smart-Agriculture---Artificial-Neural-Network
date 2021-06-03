# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 15:44:49 2018

@author: user
"""
import random
c = 50
f=open("input1.txt", "w")
while(c):
    a = random.randrange(20, 50)
    print(a)
    f.write(str(a))
    c-=1
    