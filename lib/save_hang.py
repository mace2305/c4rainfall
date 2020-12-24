#!/usr/bin/env python3

"""
Taken from https://askubuntu.com/questions/41778/computer-freezing-on-almost-full-ram-possibly-disk-cache-problem
"""

import psutil, time
import tkinter as tk
from subprocess import Popen, PIPE
import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import tkinter
from tkinter import messagebox
root = tkinter.Tk()
root.withdraw()

RAM_USAGE_THRESHOLD = 96
SWAP_USAGE_THRESHOLD = 98
MAX_NUM_PROCESS_KILL = 100

def main():
    if psutil.swap_memory().percent >= SWAP_USAGE_THRESHOLD and \
        psutil.virtual_memory().percent >= RAM_USAGE_THRESHOLD:
        # Clear RAM cache
        mem_warn = "Memory usage critical: {}%\nClearing RAM Cache".\
            format(psutil.virtual_memory().percent)
        print(mem_warn)
        Popen("notify-send \"{}\"".format(mem_warn), shell=True)
        print("Clearing RAM Cache")
        print(Popen('echo 1 > /proc/sys/vm/drop_caches',
                    stdout=PIPE, stderr=PIPE,
                    shell=True).communicate())
        post_cache_mssg = "Memory usage after clearing RAM cache: {}%".format(
                            psutil.virtual_memory().percent)
        Popen("notify-send \"{}\"".format(post_cache_mssg), shell=True)
        print(post_cache_mssg)

        if psutil.virtual_memory().percent < RAM_USAGE_THRESHOLD:
            print("Clearing RAM cache saved the day")
            return
        # Kill top C{MAX_NUM_PROCESS_KILL} highest memory consuming processes.
        ps_killed_notify = ""
        for i, ps in enumerate(sorted(psutil.process_iter(),
                                      key=lambda x: x.memory_percent(),
                                      reverse=True)):
            # Do not kill root
            if ps.pid == 1:
                continue
            elif (i > MAX_NUM_PROCESS_KILL) or \
                    (psutil.virtual_memory().percent < RAM_USAGE_THRESHOLD):
                messagebox.showwarning('Killed proccess - save_hang',
                                       ps_killed_notify)
                Popen("notify-send \"{}\"".format(ps_killed_notify), shell=True)
                return
            else:
                try:
                    ps_killed_mssg = "Killed {} {} ({}) which was consuming {" \
                                     "} % memory (memory usage={})". \
                        format(i, ps.name(), ps.pid, ps.memory_percent(),
                               psutil.virtual_memory().percent)
                    ps.kill()
                    time.sleep(1)
                    ps_killed_mssg += "Current memory usage={}".\
                        format(psutil.virtual_memory().percent)
                    print(ps_killed_mssg)
                    ps_killed_notify += ps_killed_mssg + "\n"
                except Exception as err:
                    print("Error while killing {}: {}".format(ps.pid, err))
    else:
        print("Memory usage = " + str(psutil.virtual_memory().percent))
    root.update()


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as err:
            print(err)
        time.sleep(1)