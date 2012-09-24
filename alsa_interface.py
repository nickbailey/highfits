#!/usr/bin/env python

import pyaudio

import os
import numpy
import pylab

import multiprocessing
import time

import sine
from defs_measure import INPUT_RATE, OUTPUT_RATE, ALSA_BUFFER_SIZE
from defs_measure import FORMAT_ALSA, FORMAT_NUMPY, FORMAT_ALSA_OUT, FORMAT_NUMPY_OUT

DEVICE_NUMBER = 5 # TASCAM
INTERNAL_DEVICE_NUMBER = 0 # other tascam

DEBUG_SINE = False

PREPULSE = True
PREPULSE = False
PULSE_CYCLES = 6.0
PAUSE_CYCLES = 12.0
PULSES = 10


### portaudio should use plughw
os.environ['PA_ALSA_PLUGHW'] = '1'

def sinePlay(frequency_queue, stream_out, lock):
    """ Plays a mono sine wave.  The entire wave including the
        decay (if any) is precalcuated and stored in stimulus.
        Then we wait for the off by trying to acquire the given lock."""

    freq = 440.0
    ramp_time = dur = 0
    while True:
        if DEBUG_SINE:
            freq = 5.0
            dur = 1.0
            ramp_time = 0.1        
        else:
            #while not frequency_queue.empty():
            freq, dur, ramp_time = frequency_queue.get()
        # first calculate the pure sinewave.
        print "sine at %fHz for %fs" % (freq, dur)
        stimulus = numpy.sin(numpy.linspace(0.0,
                                            2.0*numpy.pi*freq*dur,
                                            dur*OUTPUT_RATE))
        # Pre-pulse (avoid heating string)
        if PREPULSE :
            window_len = int(PULSE_CYCLES*OUTPUT_RATE/freq)
            hann_window = numpy.hanning(window_len)
            pulse = hann_window * stimulus[:window_len]
            silence = numpy.zeros(PAUSE_CYCLES*OUTPUT_RATE/freq)
            for i in range(PULSES):
                stimulus = numpy.concatenate( (pulse, silence, stimulus) )
        
        # Apply final fade-out if required.        
        if ramp_time > 0 :
            for i in range(1, int(ramp_time*OUTPUT_RATE)) :
                stimulus[-i] *= float(i)/(ramp_time*OUTPUT_RATE)

        stimulus = numpy.int16(32767*stimulus)
        # Wait for it, wait for it...
        #print "Output thread ready to send %d samples" % len(stimulus)
        #print "    ending ", stimulus[-10:]
        
        stimulus = stimulus.tostring()
        lock.acquire()
        # AAAAAGH!
        
        # audio data ready now.
        """ if freq == 0:
            stream_out.write(zeros)
        elif freq < 0:
            break
        else:
            sine_osc.fill_buffer(sine_wave)
            # Fade the contents of the buffer towards zero if required
            if sine_osc.fader_samples > 0:
				for i in range(len(sine_wave)):
					sine_wave[i] *= sine_osc.fadeout/sine_osc.fader_samples
					if sine_osc.fadeout > 0:
						sine_osc.fadeout -= 1
            sine_string = sine_wave.tostring()
            stream_out.write(sine_string)"""
        stream_out.start_stream()
        stream_out.write(stimulus)
        stream_out.stop_stream()
        # Release the play lock for next time
        lock.release()
        time.sleep(dur) # should maybe do a flush or something instead
        #print "Output samples written"
        # Phew!

# Won't be using this any more.            
#def tascam_input(tascam_input_queue, alsa_stream_in, lock):
    #multiples = PROCESS_BUFFER_SIZE / ALSA_BUFFER_SIZE
    ##lock.acquire()???!!!
    ## Start the input stream?
    #while True:
        #left = numpy.empty(PROCESS_BUFFER_SIZE, dtype=FORMAT_NUMPY)
        ##right = numpy.empty(PROCESS_BUFFER_SIZE, dtype=FORMAT_NUMPY)
        #for i in range(multiples):
            #chunk = alsa_stream_in.read(ALSA_BUFFER_SIZE)
            #data = numpy.fromstring(chunk, dtype=FORMAT_NUMPY)
            #left[i*ALSA_BUFFER_SIZE:(i+1)*ALSA_BUFFER_SIZE] = data
            ##left[i*ALSA_BUFFER_SIZE:(i+1)*ALSA_BUFFER_SIZE] = data[0::2]
            ##right[i*ALSA_BUFFER_SIZE:(i+1)*ALSA_BUFFER_SIZE] = data[1::2]
        ##tascam_input_queue.put( (left, right) )
        #tascam_input_queue.put( left )
    #lock.release()
    #print "input queue put"

def record_audio(alsa_stream_in, seconds):
    num_samples = int(seconds*INPUT_RATE)
    ### critical section...
    alsa_stream_in.start_stream()
    data = alsa_stream_in.read( num_samples )
    alsa_stream_in.stop_stream()
    ### ... end critical section
    samples = numpy.fromstring(data, dtype=FORMAT_NUMPY)
    return samples


class Audio():
    def __init__(self):
        self.alsa_obj = pyaudio.PyAudio()
        self.fader_samples = 0 # No fade-out

    def get_info(self):
        #print self.alsa_obj.get_default_host_api_info()
        alsa_info = self.alsa_obj.get_host_api_info_by_type(pyaudio.paALSA)
        alsa_index = alsa_info['index']
        for i in range(alsa_info['deviceCount']):
            alsa_device_info = self.alsa_obj.get_device_info_by_host_api_device_index(
                alsa_index, i)
            #print i, alsa_device_info['name']
            print i, alsa_device_info
        print '---'
        hosts = self.alsa_obj.get_host_api_count()
        for i in range(hosts):
            print i, self.alsa_obj.get_host_api_info_by_index(i)
        print pyaudio.paFloat32
        ok = self.alsa_obj.is_format_supported(
            INPUT_RATE,
            input_device=DEVICE_NUMBER,
            #input_channels=2,
            input_channels=1,
            input_format=FORMAT_ALSA,
            output_device=INTERNAL_DEVICE_NUMBER,
            output_channels=1,
            output_format=FORMAT_ALSA_OUT,
            )
        print ok
        #exit(0)


    def close(self):
        self.alsa_obj.terminate()
    

    def begin(self, out_lock):
        # start output first
        self.alsa_stream_out = self.alsa_obj.open(
            format=FORMAT_ALSA_OUT, channels=1, rate=OUTPUT_RATE, output=True,
            frames_per_buffer=ALSA_BUFFER_SIZE,
            output_device_index=INTERNAL_DEVICE_NUMBER,
            start = False,
            )
        self.play_sine_queue = multiprocessing.Queue()
        self.sine_process = multiprocessing.Process(
            target=sinePlay,
            args=(self.play_sine_queue, self.alsa_stream_out, out_lock)
            )
        self.sine_process.start()
        self.alsa_stream_in = self.alsa_obj.open(
            #format=FORMAT_ALSA, channels=2, rate=INPUT_RATE, input=True,
            format=FORMAT_ALSA, channels=1, rate=INPUT_RATE, input=True,
            frames_per_buffer=ALSA_BUFFER_SIZE,
            input_device_index=DEVICE_NUMBER,
            start = False,
            )
        return self.play_sine_queue, None

    def end(self):
        ### schedule clean-up
        self.play_sine_queue.put( (-1, 0) )

        ### actual clean-up
        self.sine_process.join()
        #self.alsa_stream_out.stop_stream()
        self.alsa_stream_out.close()

        #self.alsa_stream_in.stop_stream()
        self.alsa_stream_in.close()


def test_run(freq_queue, audio_in_queue):
    ### testing: do something with data
    lefts = []
    #rights = []
    seconds = 1.0
    freq_queue.put( (437, 0) )
    for i in range(int(seconds*INPUT_RATE/PROCESS_BUFFER_SIZE)):
        while audio_in_queue.empty():
            time.sleep(0.1)
        while not audio_in_queue.empty():
            #left, right = audio_in_queue.get()
            left = audio_in_queue.get()
            lefts.append(left)
            #rights.append(right)
    freq_queue.put( (437, 0.5) )
    for i in range(int(0.5*INPUT_RATE/PROCESS_BUFFER_SIZE)):
        while audio_in_queue.empty():
            time.sleep(0.1)
        while not audio_in_queue.empty():
            #left, right = audio_in_queue.get()
            left = audio_in_queue.get()
            lefts.append(left)
            #rights.append(right)
    #return lefts, rights
    return lefts, None

def test_display(lefts, rights):
    ### testing: do something with data
    ### show or save data?
    left_all = numpy.array(lefts).flatten()
    #right_all = numpy.array(rights).flatten()

    import scipy.io.wavfile
    #scipy.io.wavfile.write("test-right.wav", INPUT_RATE,
    #    numpy.int32(2**31*right_all))
    scipy.io.wavfile.write("test-left.wav", INPUT_RATE,
        numpy.int32(2**31*left_all))



def test():
    audio = Audio()
    #audio = pyaudio.PyAudio()
    #audio.get_info()
    ### real

    import multiprocessing
    lock = multiprocessing.Lock()
    lock.acquire()

    # Uncomment to try and read directly (no other process)
    
    #alsa_stream_in = audio.open(
    #        #format=FORMAT_ALSA, channels=2, rate=INPUT_RATE, input=True,
    #        format=FORMAT_ALSA, channels=1, rate=INPUT_RATE, input=True,
    #        output=False,
    #        frames_per_buffer=ALSA_BUFFER_SIZE,
    #        input_device_index=DEVICE_NUMBER,
    #        start = False,
    #        )

    freq_queue, audio_in_queue = audio.begin(lock)
    alsa_stream_in = audio.alsa_stream_in
    #lefts, rights = test_run(freq_queue, audio_in_queue)
    print "start pluck now"
    samples = record_audio(alsa_stream_in, 4.0)
    print "end pluck now"
    import pylab
    pylab.plot(samples)
    pylab.show()

    #audio.end()
    #test_display(lefts, rights)
    audio.close()

if __name__ == "__main__":
    test()
    #sinePlay(None, None, None)

