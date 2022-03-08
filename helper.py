import time
class Timer:
    def __init__(self):
        self.ini = time.time()
        self.last = 0
        self.curr = 0
        
    def timeit(self):
        if self.last == 0 and self.curr == 0:
            self.last = time.time()
            self.curr = time.time()
            return 0, 0
        else:
            self.last = self.curr
            self.curr = time.time()
            return round(self.curr - self.last, 2), round(self.curr - self.ini, 2)
