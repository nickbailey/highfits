#!/usr/bin/env python

PLOT_TRIAL = False
PLOT_TRIAL = True

import time
import pylab
import numpy
import math
import os
import scipy.io.wavfile
import scipy.signal as sig
import scipy.fftpack as fft
from BackgroundPlot import BackgroundPlot

import partials

import comedi_interface
from comedi_interface import CHANNEL_RELAY as RELAY

import alsa_interface
from defs_measure import INPUT_RATE, FORMAT_NUMPY

SECONDS_PLUCK_DECAY    = 2.0
SECONDS_MEASUREMENT    = 6.0
SECONDS_WAIT_FOR_PGA   = 0.0
LOCATE_F0_ATTEMPTS     = 3
MAX_PARTIAL            = 10
MEASUREMENTS           = 3

SAVE_PATH = "/mnt/strauss/gperciva/Violin_Tester/"
INSTR     = "labVl"
STG       = "a"

def plot_file_name(modenum, trial):
    #return "%s/%s-%s_%02d(%02d)" % (SAVE_PATH, INSTR, STG, modenum, trial)
    return "%s/%s_%s_%02d_%02d_" % (SAVE_PATH, INSTR, STG, modenum, trial)

def save_pluck_data(modenum, trial, samples):
    fnb = plot_file_name(modenum, trial)
    graphname = fnb + "-measured.png"
    wavname = fnb + ".wav"
    if FORMAT_NUMPY == numpy.int16:
        scipy.io.wavfile.write(wavname, INPUT_RATE, samples)
    else:
        scipy.io.wavfile.write(wavname, INPUT_RATE, numpy.int32((2**31-1)*samples))
    os.chmod(wavname, 0666)
 
"""
def wait(seconds, audio_in_queue):
    lefts = []
    #rights = []
    for i in range(int(seconds*INPUT_RATE/PROCESS_BUFFER_SIZE)):
        while audio_in_queue.empty():
            time.sleep(0.1)
        while not audio_in_queue.empty():
            #left, right = audio_in_queue.get()
            left = audio_in_queue.get()
            lefts.append(left)
            #rights.append(right)
    #return lefts, rights
    return lefts, None
"""

def wait_left(seconds, audio_in_queue, alsa_stream_in):
    num_samples = int(seconds*INPUT_RATE)
    ### critical section...
    alsa_stream_in.start_stream()
    data = alsa_stream_in.read( num_samples )
    alsa_stream_in.stop_stream()
    ### ... end critical section
    samples = numpy.fromstring(data, dtype=FORMAT_NUMPY)
    """
    ### clear out queue
    while not audio_in_queue.empty():
    print "input queue get"
        _ = audio_in_queue.get()

    lefts = []
    for i in range(int(seconds*INPUT_RATE/PROCESS_BUFFER_SIZE)):
        while audio_in_queue.empty():
            time.sleep(0.1)
        while not audio_in_queue.empty():
            #left, _ = audio_in_queue.get()
            left = audio_in_queue.get()
        print "input queue get"
            lefts.append(left)
    samples = numpy.array(lefts).flatten()
    """
    #pylab.plot(samples)
    #pylab.show()
    return samples

def estimate_in(spectrum, bin_l, bin_h) :
    # Look for max in bin range and fit
    # We expect spectum to stop at the Nyquist frequency
    fftlen = len(spectrum)
    print ("Scanning from bin %d (%fHz) to %d (%fHz) to locate maximum" %
           (bin_l, bin_l*INPUT_RATE/fftlen, bin_h, bin_h*INPUT_RATE/fftlen))
    exam = abs(spectrum[bin_l:bin_h+1])
    
    # Peak finding requires a bin either side of the maximum value
    x = numpy.argmax(exam[1:-1])
    x = x+1
    m = exam[x]
    print "Candidate at bin %d with value %f" % (bin_l+x, m)
    print exam[x-1:x+2]
    ym1, y0, yp1 = exam[x-1:x+2]
    p = (yp1 - ym1)/(2*(2*y0 - yp1 - ym1))
    return 0.5*(bin_l+x+p)*(INPUT_RATE)/fftlen

def stimulate(f0, seconds_decay, osc_ctrl_queue, audio_in_queue,
        comedi, alsa_in_stream):
    print "Sending stimulus signal at %fHz" % f0
    # New plan is osc_ctrk_put new freq; op proc calcs samps;
    # so put a wait 5 or wotever here 2 avoid frying rs.
    osc_ctrl_queue.put( (f0, 0) )
    #_, _ = wait(1.0, audio_in_queue)
    time.sleep(1.0)
    comedi.send(RELAY, 1)
    #_, _ = wait(2.0, audio_in_queue)
    time.sleep(2.0)
    osc_ctrl_queue.put( (f0, 0.025) )
    #_, _ = wait(0.17, audio_in_queue)
    time.sleep(0.17)
    comedi.send(RELAY, 0)
    osc_ctrl_queue.put( (0, 0) )
    #l, _ = wait(0.1, audio_in_queue)
    time.sleep(0.033)
    
    #comedi.send(comedi_interface.CHANNEL_A1, 0)
    #comedi.send(comedi_interface.CHANNEL_A0, 0)

    #print "Waiting for %f seconds to skip transient" % SECONDS_WAIT_FOR_PGA
    #_, _ = wait(SECONDS_WAIT_FOR_PGA, audio_in_queue)
    if SECONDS_WAIT_FOR_PGA > 0:
    	time.sleep(SECONDS_WAIT_FOR_PGA)
    
    #print "Reading %f seconds for analysis " % seconds_decay
    pluck = wait_left(seconds_decay, audio_in_queue, alsa_in_stream)
        
    comedi.send(comedi_interface.CHANNEL_A1, 0)
    comedi.send(comedi_interface.CHANNEL_A0, 0)

    return pluck

def find_peak(f0, osc_ctrl_queue, audio_in_queue,
        comedi, serial, alsa_stream_in):
    firstsamp = 0    # Location of first sample of pluck

    fftlen = SECONDS_PLUCK_DECAY * INPUT_RATE   # Number of bins for pitch determination FFT
    fnb = plot_file_name(f0, serial)      # Base filename for plot pngs

    pluck = stimulate(f0, SECONDS_PLUCK_DECAY,
                      osc_ctrl_queue, audio_in_queue, comedi, alsa_stream_in)

    save_pluck_data(f0, serial, pluck)
    #exit(1)
    
    spectrum = fft.fft(pluck, 2*len(pluck))  #[:4*len(tofft)/2]
    fftlen = len(spectrum)
    spectrum = spectrum[:fftlen/2]
    
    sampname = fnb + "-samples.png"
    freqname = fnb + "-spect.png"
    
    if PLOT_TRIAL:
        pluckplt = BackgroundPlot(5, sampname, 
                                  range(len(pluck)), pluck, title=fnb)

    bin_l = int((f0-50)*fftlen/INPUT_RATE)
    bin_h = int((f0+50)*fftlen/INPUT_RATE)
    estimate = estimate_in(spectrum, bin_l, bin_h)
    print "\n>>>>>>>> Sent %fHz; Peak estimate: %fHz\n" % (f0, estimate)
    
    if not PLOT_TRIAL :
        time.sleep(10) # This now should happen at top of stimulate
    else:
        freqs = [float(i)*INPUT_RATE/fftlen for i in range(bin_l, bin_h+1)]
        pluckplt.waitForPlot()
        os.chmod(sampname, 0666)
        
        gtitle = "investigating %.3fHz: estimate %f" % (f0, estimate)
        freqplt = BackgroundPlot(5, freqname, freqs,
            20*numpy.log(abs(spectrum[bin_l:bin_h+1])),
            gtitle, estimate, '.-')
        freqplt.waitForPlot()
        os.chmod(freqname, 0666)
        
    return estimate

def main():
    ### init
    comedi = comedi_interface.Comedi()
    comedi.send_all(0)
    #comedi.send(comedi_interface.CHANNEL_A0, 1)
    audio = alsa_interface.Audio()
    
    audio_input_lock = multiprocessing.Lock()
    audio_output_lock = multiprocessing.Lock()
    audio_input_lock.acquire()
    audio_output_lock.acquire()
    
    # Later on, we'll say:
    #    audio_output_lock.release()
    #    sleep(some time)
    #    pluck_data = wait_left(6.0)
    #    ## audio_input_lock.release()
    #    # get some data
    #    # When everything settles down, both processes should
    #    # release their locks (might need some extra logic in them)
    #    audio_input_lock.acquire()
    #    audio_output_lock.acquire()
    #    queue.send(next buffer of stuff)
    # as many times as is necessary,
    # and this should play the whole precomputed output block
    # starting the input some time into it.
    # Seems the separate output thread tascam_input_process
    # is deprecated in alsa_interface.py, so need some consultancy
    # on whether this is what you're thinking about.
    # Also need to precompute the blocks instead of using the
    # freq_queue stuff which might be a bit of a big change.
    
    freq_queue, audio_in_queue = audio.begin(audio_input_lock)
    
    alsa_stream_in = audio.alsa_stream_in

    f0=438.0         # Expected frequency of fundamental
    B = 0.0
    tries = LOCATE_F0_ATTEMPTS
    max_partials = MAX_PARTIAL
    
    series = [0] * (max_partials+1)

    tested_modes = []
    tested_favgs = []
    for p in range(1, max_partials+1):
        outfile = open("estimates-%i.txt" % p, "w")
        f_test = partials.mode_B2freq(f0, p, B)
        f_avg = 0
        for attempt in range(tries):
            f_measured = find_peak(f_test, freq_queue, audio_in_queue,
                comedi, attempt, alsa_stream_in)
            f_avg += f_measured
            f_test = 0.1*f_test + 0.9*f_measured
            outfile.write("%f\n" % f_measured)
            #time.sleep(10)
        f_avg /= tries
        tested_modes.append(p)
        tested_favgs.append(f_avg)
        f0, B, rsquared = partials.estimate_B(
            tested_modes, tested_favgs, f0, B)
        print "--------------"
        print "%.2f\t%.3g\t%.3g" % (f0, B, rsquared)
        print "--------------"
        #f0 = f0avg * (p + 1) / p
        series[p] = f_avg
        outfile.close()
        
    series = series[1:]
    
    # High pass de-glitch filter
    dgfa, dgfb = sig.iirdesign(50.*0.5/INPUT_RATE, 20*0.5/INPUT_RATE, 3, 70)
    numpy.savetxt( os.path.join(SAVE_PATH, "detected-freqs.txt"),
        numpy.array(series) )

    for f0_idx in range(len(series)) :
        f0 = series[f0_idx]
        for attempt in range(MEASUREMENTS) :
            pluck = stimulate(f0, SECONDS_MEASUREMENT,
                              freq_queue, audio_in_queue,
                              comedi, alsa_stream_in)
            save_pluck_data(f0_idx+1, attempt, pluck)

            if PLOT_TRIAL:
                fnb = plot_file_name(f0_idx+1, attempt)
                graphname = fnb + "-measured.png"
                pluckplt = BackgroundPlot(10, graphname, 
                                          range(len(pluck)), pluck,
                                          title=fnb)
                pluckplt.waitForPlot()
                os.chmod(graphname, 0666)

            else:
                time.sleep(10)

    ### clean up
    audio.end()
    audio.close()


if __name__ == "__main__":
    main()

