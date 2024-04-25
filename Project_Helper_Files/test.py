import os
import time
import tkinter as tk
import threading

name_list = ['',"Alex Camaj"]
def GetName():
    return name_list

def AddName(string):
    name_list.append(str(string))


id = len(GetName())-1
print(id)