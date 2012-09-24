""" A wire between input and output. """

import pyaudio
import sys

chunk = 4096+0
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

p = pyaudio.PyAudio()

stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                output = True,
                frames_per_buffer = chunk)

print "* recording"
for i in range(0, 44100 / chunk * RECORD_SECONDS):
    data = stream.read(chunk)
    stream.write(data, chunk,exception_on_underflow=True)
    for i in range(9000):
        print i
print "* done"

stream.stop_stream()
stream.close()
p.terminate()

