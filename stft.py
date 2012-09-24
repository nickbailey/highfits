#!/usr/bin/env python
import numpy
import scipy.io.wavfile
import scipy.fftpack
import scipy.signal


### convenience functions
def amplitude2db(power):
    return 20.0 * scipy.log10( power )

def db2amplitude(db):
    return 10.0**(db/20.0)

def hertz2bin(freq, sample_rate, WINDOWSIZE):
    return freq*(WINDOWSIZE/2+1) / (float(sample_rate)/2)

def bin2hertz(bin_number, sample_rate, WINDOWSIZE):
    return bin_number * (sample_rate/2) / (float(WINDOWSIZE)/2+1)

