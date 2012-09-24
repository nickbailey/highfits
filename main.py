#!/usr/bin/env python

import time

import numpy
import scipy.io.wavfile

import comedi_interface
from comedi_interface import CHANNEL_RELAY as RELAY

import alsa_interface
from defs_measure import RATE, PROCESS_BUFFER_SIZE, FORMAT_NUMPY

def wait(seconds, audio_in_queue):
    lefts = []
    rights = []
    for i in range(int(seconds*RATE/PROCESS_BUFFER_SIZE)):
        while audio_in_queue.empty():
            time.sleep(0.1)
        while not audio_in_queue.empty():
            left, right = audio_in_queue.get()
            lefts.append(left)
            rights.append(right)
    return lefts, rights




def main():
    ### init
    comedi = comedi_interface.Comedi()
    comedi.send_all(0)
    comedi.send(comedi_interface.CHANNEL_A0, 1)
    audio = alsa_interface.Audio()
    freq_queue, audio_in_queue = audio.begin()

    #freq_queue.put(0)
    
    if True:
        #freq_queue.put( (437.8, 0) )
        freq_queue.put( (300.0, 0) )
        _, _ = wait(0.5, audio_in_queue) # better safe than sorry
        comedi.send(RELAY, 1)
        power_lefts = []
        power_rights = []
        for i in range(0, 20) :
            f_out = 439 + i/10.0
            print f_out
            freq_queue.put( (f_out, 0) )
            #power_lefts, power_rights = wait(5.0, audio_in_queue)
            l, r = wait(5.0, audio_in_queue)
            power_lefts += l
            power_rights += r
		# prepare to turn off
        #freq_queue.put( (437, -0.25) )
        #turn_lefts, turn_rights = wait(0.3, audio_in_queue)

        # turn off signal
        freq_queue.put( (0, 0) )
        turn_lefts, turn_rights = wait(0.1, audio_in_queue)
        comedi.send(RELAY, 0)

        turn_lefts2, turn_rights2 = wait(0.1, audio_in_queue)
        comedi.send(comedi_interface.CHANNEL_A0, 1)
        #comedi.send(comedi_interface.CHANNEL_A0, 1)
        
        decay_lefts, decay_rights = wait(5.0, audio_in_queue)
        freq_queue.put( (-1, 0) )
    
    ### clean up
    audio.end()
    audio.close()
    
    left_all = numpy.array(power_lefts + turn_lefts + turn_lefts2 + decay_lefts).flatten()
    right_all = numpy.array(power_rights + turn_rights + turn_rights2 + decay_rights).flatten()
    #left_all = numpy.array(power_lefts + turn_lefts + decay_lefts).flatten()
    #right_all = numpy.array(power_rights + turn_rights + decay_rights).flatten()
    #left_all = numpy.array(power_lefts + decay_lefts).flatten()
    #right_all = numpy.array(power_rights + decay_rights).flatten()
    
    #left_all = numpy.array(power_lefts).flatten()
    #right_all = numpy.array(power_rights).flatten()
    #left_all = numpy.append(left_all, numpy.array(decay_lefts).flatten())
    #right_all = numpy.append(right_all, numpy.array(decay_rights).flatten())
    
    if FORMAT_NUMPY == numpy.int16:
        scipy.io.wavfile.write("test-left.wav", RATE,
            left_all)
        scipy.io.wavfile.write("test-right.wav", RATE,
            right_all)
        scipy.io.wavfile.write("decay.wav", RATE,
            numpy.array(decay_lefts).flatten())
    else:
        scipy.io.wavfile.write("test-left.wav", RATE,
            numpy.int32((2**31-1)*left_all))
        scipy.io.wavfile.write("test-right.wav", RATE,
            numpy.int32((2**31-1)*right_all))
        scipy.io.wavfile.write("decay.wav", RATE,
            numpy.int32((2**31-1)*decay_lefts))


if __name__ == "__main__":
    main()

