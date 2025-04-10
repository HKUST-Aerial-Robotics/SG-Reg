import time

class TicToc:
    def __init__(self):
        self.tic()
        
    def tic(self):
        self.t0 = time.time()
        
    def toc(self):
        duration = time.time() - self.t0
        self.tic()
        return duration
    
