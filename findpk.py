#!/usr/bin/env python

PLOT_TRIAL = False
#PLOT_TRIAL = True

import time
import pylab
import numpy
import math
import os
import multiprocessing
import scipy.io.wavfile
import scipy.signal as sig
import scipy.fftpack as fft
from BackgroundPlot import BackgroundPlot

import partials
import spectral_subtraction

import comedi_interface
from comedi_interface import CHANNEL_RELAY as RELAY

import alsa_interface
from defs_measure import INPUT_RATE, OUTPUT_RATE, FORMAT_NUMPY

import chemistry
import estimate_f0_B

SECONDS_PEAKFINDER_FFT = 20.0
SECONDS_MEASUREMENT    = 3.0 # Original setting
SECONDS_MEASUREMENT    = 1.5
SECONDS_MEASUREMENT = SECONDS_PEAKFINDER_FFT

SECONDS_WAIT_FOR_PGA   = 0.0
SECONDS_STIMULUS       = 0.8 # A string
SECONDS_STIMULUS       = 0.3 # E
SECONDS_STIMULUS       = 0.5 # cello D

#SECONDS_RELAY_RELEASE  = 0.017
SECONDS_RELAY_RELEASE  = 0.02
LOCATE_F0_ATTEMPTS     = 20
MAX_PARTIAL            = 1
MEASUREMENTS           = 2

LOGFILE   = "info-findpk.log"
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

def wait_left(seconds, alsa_stream_in):
    num_samples = int(seconds*INPUT_RATE)
    ### critical section...
    alsa_stream_in.start_stream()
    data = alsa_stream_in.read( num_samples )
    alsa_stream_in.stop_stream()
    ### ... end critical section
    samples = numpy.fromstring(data, dtype=FORMAT_NUMPY)

    #pylab.plot(samples)
    #pylab.show()
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
    #print ("Scanning from bin %d (%fHz) to %d (%fHz) to locate maximum" %
    #       (bin_l, bin_l*INPUT_RATE/fftlen, bin_h, bin_h*INPUT_RATE/fftlen))
    exam = abs(spectrum[bin_l:bin_h+1])
    
    # Peak finding requires a bin either side of the maximum value
    x = numpy.argmax(exam[1:-1])
    x = x+1
    m = exam[x]
    #print "Candidate at bin %d with value %f" % (bin_l+x, m)
    #print exam[x-1:x+2]
    ym1, y0, yp1 = exam[x-1:x+2]
    p = (yp1 - ym1)/(2*(2*y0 - yp1 - ym1))
    yp = y0 - 0.25*(ym1-yp1)*p
    estimated_bin = 0.5*(bin_l+x+p)*(INPUT_RATE)/fftlen
    estimated_mag = yp
    return estimated_bin, estimated_mag

def stimulate(f0, seconds_decay,
              osc_ctrl_queue, lock,
              audio_in_queue,
              comedi, alsa_in_stream):
    #print "Sending stimulus signal at %fHz" % f0
    # New plan is osc_ctrl_put new freq; op proc calcs samps;
    # so put a wait 5 or wotever here 2 avoid frying rs.
    
    # Play a sine wave at f0 for 0.75 seconds,
    # fading to 0 over the last 0.025 seconds.
    osc_ctrl_queue.put( (f0, SECONDS_STIMULUS, 0.025) )
    # The output thread will start to calculate the audio samples now.
    # While the CPU heats up, we let the resistors cool down.
    if PLOT_TRIAL : # Plotting all those graphs took a while
        time.sleep(0.5)
    else :
        time.sleep(9.7)
    comedi.send(RELAY, 1)
    time.sleep(0.3)
    # Set the output running
    #print "releasing output lock"
    lock.release()
    
    if alsa_interface.PREPULSE :
        total_stimulus_time = SECONDS_STIMULUS + ( alsa_interface.PULSES *
            (alsa_interface.PULSE_CYCLES + alsa_interface.PAUSE_CYCLES) / f0)
    else:
        total_stimulus_time = SECONDS_STIMULUS
    
    time.sleep(total_stimulus_time-SECONDS_RELAY_RELEASE)  # while most of the output
    comedi.send(RELAY, 0)
    time.sleep(SECONDS_RELAY_RELEASE) # Relay goes clunk
    comedi.send(comedi_interface.CHANNEL_A1, 0)
    comedi.send(comedi_interface.CHANNEL_A0, 1)    
    #comedi.send(comedi_interface.CHANNEL_A0, 0)    
    time.sleep(SECONDS_WAIT_FOR_PGA)  # PGA settles down    
    # Read the input
    #alsa_in_stream.start_stream()
    pluck= wait_left(seconds_decay, alsa_in_stream)
    # read everything (6.0 seconds?)
    #alsa_in_stream.stop_stream()
    
    
    """osc_ctrl_queue.put( (f0, 0) )
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
    time.sleep(0.033)"""
    
    #print "Waiting for %f seconds to skip transient" % SECONDS_WAIT_FOR_PGA
    #_, _ = wait(SECONDS_WAIT_FOR_PGA, audio_in_queue)
    #if SECONDS_WAIT_FOR_PGA > 0:
   #    time.sleep(SECONDS_WAIT_FOR_PGA)
    
    #print "Reading %f seconds for analysis " % seconds_decay
    #pluck = wait_left(seconds_decay, audio_in_queue, alsa_in_stream)
     
     
    # general end clean-up   
    comedi.send(comedi_interface.CHANNEL_A1, 0)
    comedi.send(comedi_interface.CHANNEL_A0, 0)

    # Put the brakes on the output process
    lock.acquire()
    return pluck

def find_peak(f0, osc_ctrl_queue, lock, audio_in_queue,
        comedi, serial, alsa_stream_in):
    firstsamp = 0    # Location of first sample of pluck

    fftlen = SECONDS_PEAKFINDER_FFT * INPUT_RATE   # Number of bins for pitch determination FFT
    fnb = plot_file_name(f0, serial)      # Base filename for plot pngs

    pluck = stimulate(f0, SECONDS_PEAKFINDER_FFT,
                      osc_ctrl_queue, lock, audio_in_queue,
                      comedi, alsa_stream_in)

    save_pluck_data(f0, serial, pluck)

    #data_un_silenced = spectral_subtraction.spectral_subtraction_arrays(
    #    pluck, silence, INPUT_RATE)

    fft_points = 2*len(pluck)
    spectrum = fft.fft(pluck, fft_points)  #[:4*len(tofft)/2]
    fftlen = len(spectrum)
    spectrum = spectrum[:fftlen/2+1] / fft_points

    power_spectrum = abs(spectrum)**2
    
    sampname = fnb + "-samples.png"
    freqname = fnb + "-spect.png"
    
    if PLOT_TRIAL:
        pluckplt = BackgroundPlot(5, sampname, 
                                  range(len(pluck)), pluck, title=fnb)
        #pylab.plot(pluck, '.-')
        #pylab.show()

    #freq_spread = 50.0
    freq_spread = 5.0
    bin_l = int((f0 - freq_spread)*fftlen/INPUT_RATE)
    bin_h = int((f0 + freq_spread)*fftlen/INPUT_RATE)
    bin_f0 = int((f0)*fftlen/INPUT_RATE)

    estimate_bin, estimate_mag = estimate_in(spectrum, bin_l, bin_h)
    estimate = estimate_bin

    #estimate, decay, estimate_mag, _, _ = chemistry.fit_lorentzian(power_spectrum, 
    #    bin_l, bin_h, bin_f0,
    #    False, False)
    #    #True, False)

    peak_dB = 10*numpy.log10(estimate_mag)
    decay = -1

    #mean_spectral_energy = abs(spectrum).mean()
    #peak_energy_above_mean = (
    #        20*numpy.log10(estimate_mag) -
    #        20*numpy.log10(mean_spectral_energy))
    #print "\n>>>>>>>> Sent %fHz; Peak estimate: %fHz\n" % (f0, estimate)
    text = "Sent %.2fHz; Peak estimate: %.2fHz; Peak magnitude: %.2f dB; Decay rate: %.3f" % (f0, estimate, peak_dB, decay)
    logfile.write(text+"\n")
    logfile.flush()
    
    if PLOT_TRIAL :
        freqs = [float(i)*INPUT_RATE/fftlen for i in range(bin_l, bin_h+1)]
        pluckplt.waitForPlot()
        os.chmod(sampname, 0666)
        
        gtitle = "investigating %.3fHz: estimate %f" % (f0, estimate)
        freqplt = BackgroundPlot(5, freqname, freqs,
            20*numpy.log10(abs(spectrum[bin_l:bin_h+1])),
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
    
    audio_output_lock = multiprocessing.Lock()
    audio_output_lock.acquire()

    global logfile
    logfile = open(LOGFILE, 'w')
    

    # Later on, we'll say (in stimulate(), actually):
    #    audio_output_lock.release()
    #    sleep(some time)
    #    pluck_data = wait_left(...
    # and this should play the whole precomputed output block
    # starting the input some time into it.
    
    freq_queue, audio_in_queue = audio.begin(audio_output_lock)
    
    alsa_stream_in = audio.alsa_stream_in

    ### get silence
    if False:
        global silence
        print "getting silence..."
        silence = stimulate(0.0, 1*SECONDS_MEASUREMENT,
                            freq_queue, audio_output_lock,
                            audio_in_queue,
                            comedi, alsa_stream_in)
        print "... silence gotten"
        wavname = "silence.wav"
        if FORMAT_NUMPY == numpy.int16:
            scipy.io.wavfile.write(wavname, INPUT_RATE, silence)
        else:
            scipy.io.wavfile.write(wavname, INPUT_RATE,
                numpy.int32((2**31-1)*silence))

    #f0=438.0         # Expected frequency of fundamental
    #f0=658.0         # Expected frequency of fundamental
    #f0=2*658.0
    #f0= .0         # Expected frequency of fundamental
    #f0 = 143.0
    f0 = 220.0
    B = 0.0
    tries = LOCATE_F0_ATTEMPTS
    max_partials = MAX_PARTIAL
    
    series = [0] * (max_partials+1)

    start_partial = 1
    tested_modes = []
    tested_favgs = []
    ### kick-start process
    #start_partial = 10
    #tested_modes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #tested_favgs = [  439.69665405  , 879.34497282  ,1321.67762859,  1763.11055972 , 2204.01194094 ,
    #  2649.08665505 , 3094.1608183 ,  3533.91024435 , 3986.87554256]
    for p in range(start_partial, max_partials+1):
        text = "--- mode %i" % (p)
        logfile.write(text+"\n")
        logfile.flush()

        outfile = open("estimates-%i.txt" % p, "w")
        f0, B, rsquared = estimate_f0_B.estimate_B(
            tested_modes, tested_favgs, f0, B)
        f_test = partials.mode_B2freq(f0, p, B)
        f_avg = 0
        for attempt in range(tries):
            f_measured = find_peak(f_test,
                                   freq_queue, audio_output_lock,
                                   audio_in_queue,
                                   comedi, attempt, alsa_stream_in)
            f_avg += f_measured
            f_test = 0.1*f_test + 0.9*f_measured
            outfile.write("%f\n" % f_measured)
            #time.sleep(10)
        f_avg /= tries # no, not useful
        tested_modes.append(p)
        tested_favgs.append(f_avg)
        if p == 1:
            f0 = f_test
        f0, B, rsquared = estimate_f0_B.estimate_B(
            tested_modes, tested_favgs, f0, B)
        #text = "Estimated string f0: %.2f\tinharmonicity B: %.3g\tR^2 \"variability\": %.3g" % (f0, B, rsquared)
        text = "Estimated string f0: %.2f\tinharmonicity B: %.3g\tR^2: %.3g" % (f0, B, rsquared)
        logfile.write(text+"\n")
        logfile.flush()
        #f0 = f0avg * (p + 1) / p
        series[p] = f_avg
        outfile.close()
        
    series = series[1:]
    
    # High pass de-glitch filter
    # dgfa, dgfb = sig.iirdesign(50.*0.5/INPUT_RATE, 20*0.5/INPUT_RATE, 3, 70)
    # (but we still save the raw data)
    numpy.savetxt( os.path.join(SAVE_PATH, "detected-freqs.txt"),
        numpy.array(series) )
    exit(1)

    for f0_idx in range(len(series)) :
        f0 = series[f0_idx]
        for attempt in range(MEASUREMENTS) :
            pluck = stimulate(f0, SECONDS_MEASUREMENT,
                              freq_queue, audio_output_lock,
                              audio_in_queue,
                              comedi, alsa_stream_in)
            save_pluck_data(f0_idx+1, attempt, pluck)

            if PLOT_TRIAL:
                fnb = plot_file_name(f0_idx+1, attempt)
                graphname = fnb + "-measured.png"
                pluckplt = BackgroundPlot(7, graphname, 
                                          range(len(pluck)), pluck,
                                          title=fnb)
                pluckplt.waitForPlot()
                os.chmod(graphname, 0666)

    ### clean up
    audio.end()
    audio.close()
    logfile.close()


if __name__ == "__main__":
    main()

