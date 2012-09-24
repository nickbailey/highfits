#!/usr/bin/env python

import numpy
import pylab

from defs_measure import FORMAT_NUMPY_OUT

SIZE = 2**16
#SIZE = 2**24

class SineInt():
    def __init__(self, freq=1.0, sample_rate=44100):
        if FORMAT_NUMPY_OUT == numpy.int16:
            self.multiplier = float(SIZE/4 - 1)
        else:
            self.multiplier = 0.5
        self.lookup = FORMAT_NUMPY_OUT(
            self.multiplier * numpy.sin(numpy.linspace(0,
                2*numpy.pi, SIZE)))
        #print self.lookup.dtype
        self.index = 0.
        self.index_advance = 0.
        self.sample_rate = sample_rate
        self.fadeout = 0.
        self.fader_samples = 0.

        self.set_freq(float(freq))
        
        #pylab.plot(self.lookup)
        #pylab.show()

    def set_freq(self, freq):
        self.index_advance = freq * SIZE / self.sample_rate

    def fill_buffer(self, buf):
        # TODO: speed this up
        for i in range(len(buf)):
            #buf[i] = self.lookup[int(self.index)]
            buf[i] = self.lookup[self.index]
            self.index += self.index_advance
            if self.index >= SIZE:
                self.index -= SIZE
            

def test():
    a = SineInt(437.0, 44100 )
    #buf = numpy.zeros(8)
    buf = numpy.zeros( 1000, dtype=FORMAT_NUMPY_OUT )
    #print buf.dtype
    a.fill_buffer(buf)
    longbuf = buf

    a.set_freq( 437 )
    buf = numpy.zeros( 4000, dtype=FORMAT_NUMPY_OUT )
    a.fill_buffer(buf)
    longbuf = numpy.append(longbuf, buf)
    
    a.set_freq( 437 )
    buf = numpy.zeros( 5000, dtype=FORMAT_NUMPY_OUT )
    a.fill_buffer(buf)
    longbuf = numpy.append(longbuf, buf)

    pylab.plot(longbuf)
    pylab.show()

if __name__ == "__main__":
    test()


