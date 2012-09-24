import threading
import matplotlib.pyplot
import time

class BackgroundPlot :
    def __init__(self, delay, fname, x, y, title=None, line=None, *args) :
        self.delay = delay
        self.plot_file_name = fname
        self.start_time = time.time()
        self.reason = None # default - terminates naturally (no user interaction)
        self.t = threading.Thread(target=self.doPlt, args=(x, y, title, line, args))
        self.t.start()

    def doPlt(self, x, y, title, line, args):
        theFigure = matplotlib.pyplot.figure()
        axes = theFigure.add_axes([.1,.1,.8,.8])
        if title != None:
            axes.set_title(title)
        if line != None:
            axes.axvline(line, color='r')
        axes.plot(x, y, *args)
        self.reason = theFigure.waitforbuttonpress(timeout=self.delay)
        theFigure.savefig(self.plot_file_name)
        matplotlib.pyplot.close(theFigure)
        
    def waitForPlot(self):
        self.t.join()
        self.t = None
        if self.reason != None : # mouse or key pressed
            remaining = time.time() - self.start_time
            if remaining > 0 :
                time.sleep(remaining)  # better safe than sorry

if __name__ == '__main__' :
    plt1 = BackgroundPlot(4, [1,2,3,4], [6,7,8,9], line=2.5)
    plt1.waitForPlot()
    plt2 = BackgroundPlot(6, [1,2,3,4], [9,8,7,6], 'Second graph', None, '.-')
    plt2.waitForPlot()
#    time.sleep(25)

