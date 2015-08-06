import os
import time

from printing import *


class Timer:
    timer_name = ""
    timer_start = 0
    sleep_start = 0
    color = bcolors.LIGHT_PURPLE

    def __init__(self, _name="", _color=bcolors.LIGHT_PURPLE):
        self.timer_name = _name
        self.color = _color

    def start(self):
        self.timer_start = time.time()
        print_f("Starting...", color=bcolors.LIGHT_PURPLE, field_name=self.timer_name)

    def pause(self):
        self.sleep_start = time.time()
        print_f("Pausing...", color=bcolors.LIGHT_PURPLE, field_name=self.timer_name)

    def resume(self):
        if self.sleep_start == 0 or self.timer_start == 0:
            return
        self.timer_start += time.time() - self.sleep_start
        self.sleep_start = 0
        print_f("Resuming...", color=bcolors.LIGHT_PURPLE, field_name=self.timer_name)

    def now(self):
        timer_result = time.time() - self.timer_start
        if self.sleep_start != 0:
            timer_result -= time.time() - self.sleep_start

        print_f(timer_result, color=bcolors.LIGHT_PURPLE, field_name=self.timer_name)

    def stop(self):
        if self.timer_start == 0:
            return
        timer_result = time.time() - self.timer_start

        self.timer_start = 0
        self.sleep_start = 0

        print_f("Stopping...", color=bcolors.LIGHT_PURPLE, field_name=self.timer_name)
        print_f(timer_result, color=bcolors.LIGHT_PURPLE, field_name=self.timer_name)