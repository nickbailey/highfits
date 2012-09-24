#!/usr/bin/env python

import sys

sys.path.append('../build/swig')
sys.path.append('../build/.libs')

import os
import datetime
import time

import violin_instrument
import monowav

import numpy
import pylab
import scipy.stats
import scipy.fftpack

import random
import aubio.aubiowrapper

import expected_frequencies
import published_constants

import arff

WINDOWSIZE = 2048
HOPSIZE = 2048
TEST_SECONDS = 1.0
SAVE_DIRNAME = "pc-consts"

# higher values are more noisy, lower values are more "spiky"
#MAX_SPECTRAL_FLATNESS = 0.01
MAX_SPECTRAL_FLATNESS = 0.5

NUM_PITCH_AVERAGE = 5

PC_CONSTS_FORMAT = """{
  %(T).1f, // T
  %(L).3f, / l
  %(d).2e, // d
  %(pl).2e, // pl
  %(E).2e, // E
  %(B1).2f, %(B2).2f, // B1, B2
}"""

import inspect
def props(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr

class Inst():
    def __init__(self, inst_num, inst_text, st, st_text):
        self.pitch_obj = aubio.aubiowrapper.new_aubio_pitchdetection(
            WINDOWSIZE, HOPSIZE, 1, 44100,
            aubio.aubiowrapper.aubio_pitch_yinfft,
            aubio.aubiowrapper.aubio_pitchm_freq,
            )
        self.fvec = aubio.aubiowrapper.new_fvec(HOPSIZE, 1)
        self.vln = violin_instrument.ViolinInstrument(inst_num)
        self.audio_out_buf = monowav.shortArray(HOPSIZE)
        self.force_out_buf = monowav.shortArray(HOPSIZE/4)

        self.inst_text = inst_text
        self.st = st
        self.st_text = st_text

        # icky!
        self.friction = {}

        self.attempts = []

    def save_consts(self):
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if not os.path.exists(SAVE_DIRNAME):
            os.makedirs(SAVE_DIRNAME)
        basename = os.path.join(SAVE_DIRNAME, date)
        while os.path.exists(basename+".txt"):
            time.sleep(1)
            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            basename = os.path.join(SAVE_DIRNAME, date)
        self.write_constants(basename)
        self.test_bow(basename, 1.0)

    def write_constants(self, basename):
        out = open(basename + ".txt", 'w')
        pc = self.vln.get_physical_constants(self.st)
        p = props(pc)
        out.write(PC_CONSTS_FORMAT % p)
        out.write(
            "\n{%.2f, %.2f, %.2f}" % (self.friction['s'],
            self.friction['d'], self.friction['v0']))
        out.close()

    def pick_new_constants(self):
        pc_ideal = published_constants.PHYSICAL_CONSTANT_RANGES[self.inst_text][self.st_text]
        pc = self.vln.get_physical_constants(self.st)
        for key in pc_ideal:
            low, high = pc_ideal[key]
            new_value = random.uniform(low, high)
            setattr(pc, key, new_value)
        f_target = expected_frequencies.FREQS[self.inst_text][self.st_text]
        print "tension correction for %.1f: old %.1f N\t" % (
            f_target, pc.T),
        pc.T = 4* f_target**2 * pc.pl * pc.L**2
        print "new %.1f N" % (pc.T)
        #pc.N = 32
        self.vln.set_physical_constants(self.st, pc)

    def pick_new_friction(self):
        friction_ideal = published_constants.FRICTION_CHARACTERISTICS[self.inst_text][self.st_text]
        friction = {}
        for key in friction_ideal:
            low, high = friction_ideal[key]
            new_value = random.uniform(low, high)
            #setattr(friction, key, new_value)
            friction[key] = new_value
        self.vln.set_friction_constants(self.st,
            friction['s'], friction['d'], friction['v0']);
        self.friction = friction

    def schelleng_max(self, beta, vb):
        pc = self.vln.get_physical_constants(self.st)
        Zc = numpy.sqrt(pc.T * pc.pl)
        print Zc
        us = self.friction['s']
        ud = self.friction['d']
        v0 = self.friction['v0']
        force_max = vb * 2*Zc / (beta*(us-ud))
        print force_max
        force_max = 2*Zc/(us-ud) * (vb + beta*v0) / beta
        print force_max
        print '---'
        return force_max

    def new_constants(self):
        stable = False
        while not stable:
            #print "picking new constants"
            self.pick_new_constants()
            self.pick_new_friction()
            beta = 0.2
            vb = 0.5
            bow_force = self.schelleng_max(beta=beta, vb=vb)
            stable = self.test_stable(beta=beta,
                bow_force=bow_force/10, vb=vb)
            if stable:
                f_target = expected_frequencies.FREQS['violin'][self.st_text]
                stable = self.tune(f_target)
            self.add_arff(stable)

    def hop_pitch(self):
        stable = self.vln.wait_samples_safe(self.audio_out_buf, HOPSIZE)
        if not stable:
            return False
        for i in xrange(HOPSIZE):
           aubio.aubiowrapper.fvec_write_sample(
               self.fvec, float(self.audio_out_buf[i]), 0, i)
        self.pitches[self.pi] = aubio.aubiowrapper.aubio_pitchdetection(
            self.pitch_obj, self.fvec)
        self.pi = (self.pi + 1) % NUM_PITCH_AVERAGE
        return True

    def tune(self, f_target):
        self.pitches = numpy.zeros(NUM_PITCH_AVERAGE)
        self.pi = 0

        K_p = 0.01
        bow_force = 0.5
        INITIAL_SECONDS = 1.0
        self.vln.bow(self.st, 0.12, bow_force, 0.25)

        # start string vibrating
        for i in range(int(INITIAL_SECONDS * 44100 / HOPSIZE)):
            stable = self.hop_pitch()
            if not stable:
                return False
            #print "%.1f" % (pitch)
        while True:
            for i in xrange(NUM_PITCH_AVERAGE):
                stable = self.hop_pitch()
                if not stable:
                    return False
                pitch = numpy.median(self.pitches)
                #print self.pitches
                #print "%.1f" % (pitch)
            if pitch < 10:
                print "unstable"
                return False
                #raise Exception("Invalid pitch")
            delta_pitch = f_target - pitch
            if abs(delta_pitch) / f_target < 0.001:
                #print "%.1f\t%.1f" % (
                #    pitch, delta_pitch)
                break
            pc = self.vln.get_physical_constants(self.st)
            pc.T += K_p * delta_pitch
            #print "%.1f\t%.1f\t%.1f" % (
            #    pitch, delta_pitch, pc.T)
            self.vln.set_physical_constants(self.st, pc)
        return True

    @staticmethod
    def spectral_flatness(memory_buf):
        buf = numpy.empty(HOPSIZE)
        for i in xrange(HOPSIZE):
            buf[i] = memory_buf[i]
        if all(buf == 0):
            return False
        fft = scipy.fftpack.fft(buf)
        fft_power = abs(fft[:len(fft)/2])**2
        flatness = (scipy.stats.gmean(fft_power) / fft_power.mean() )
        #pylab.semilogy(fft_power)
        #pylab.show()
        return flatness

    @staticmethod
    def spectral_forces(memory_buf):
        buf = numpy.empty(HOPSIZE/4)
        for i in xrange(HOPSIZE/4):
            buf[i] = memory_buf[i]
        fft = scipy.fftpack.fft(buf)
        fft_power = abs(fft[:len(fft)/2])**2
        freqs = numpy.array([ i*11025/(HOPSIZE/4)
            for i in range(len(fft_power))] )
        #pylab.figure()
        #pylab.plot(buf)
        pylab.figure()
        pylab.semilogy(freqs, fft_power)
        pylab.show()
        return True


    def test_stable(self, beta, bow_force, vb):
        self.vln.reset()
        # quick test
        wavfile = monowav.MonoWav("foo.wav")
        for i in range(int(TEST_SECONDS * 44100 / HOPSIZE)):
            self.vln.bow(self.st, beta, bow_force, vb)
            buf = wavfile.request_fill(HOPSIZE)
            #stable = self.vln.wait_samples_safe(self.audio_out_buf, HOPSIZE)
            stable = self.vln.wait_samples_safe(buf, HOPSIZE)
            if stable > 0:
                print stable
                print "bail stable 1"
                print bow_force
                exit(1)
                return False
            #self.vln.wait_samples_forces(self.audio_out_buf,
            #    self.force_out_buf, HOPSIZE)
            #if not stable:
            #    return False
        print 'ok'
        exit(1)
        stable = self.vln.wait_samples_safe(self.audio_out_buf, HOPSIZE)
        #print stable
        if stable > 0:
            print "bail stable 2"
            return False
        #self.vln.wait_samples_forces(self.audio_out_buf,
        #   self.force_out_buf, HOPSIZE)
        #if not stable:
         #   return False
        flatness = self.spectral_flatness(self.audio_out_buf)
        #force_ok = self.spectral_forces(self.force_out_buf)
        if not flatness:
            print "bail flatness 1"
            return False
        #print flatness
        if flatness > MAX_SPECTRAL_FLATNESS:
            print "bail flatness 2"
            return False
        return True

    def test_pluck(self, seconds=1.0):
        num_samples = int(seconds*44100.0)
        print "not implemented"
        #buf = self.wavfile.request_fill(num_samples)
        #self.vln.pluck(self.st, 0.48, 1.0)
        #self.vln.wait_samples(buf, num_samples)

    def test_bow(self, basename, seconds=1.0):
        self.vln.reset()
        self.wavfile = monowav.MonoWav(basename + ".wav")
        num_samples = int(seconds*44100.0)
        #self.vln.bow(self.st, 0.12, 0.33, 0.25)
        self.vln.bow(self.st, 0.12, 0.1, 0.2)
        for i in range(int(num_samples / HOPSIZE)):
            buf = self.wavfile.request_fill(HOPSIZE)
            self.vln.wait_samples(buf, HOPSIZE)

    def add_arff(self, stable):
        pc_arff = self.vln.get_physical_constants(self.st)
        pc_ideal = published_constants.PHYSICAL_CONSTANT_RANGES[self.inst_text][self.st_text]
        for key in pc_ideal:
            low, high = pc_ideal[key]
            normalized_value = (getattr(pc_arff, key) - low) / (high-low)
            setattr(pc_arff, key, normalized_value)
        friction_ideal = published_constants.FRICTION_CHARACTERISTICS[self.inst_text][self.st_text]
        friction_normalized = dict(self.friction)
        for key in friction_ideal:
            low, high = friction_ideal[key]
            new_value = (self.friction[key] - low) / (high-low)
            #setattr(friction, key, new_value)
            friction_normalized[key] = new_value
        self.attempts.append( [
            pc_arff.T, pc_arff.L, pc_arff.d, pc_arff.pl, pc_arff.E, pc_arff.B1, pc_arff.B2,
            friction_normalized['s'], friction_normalized['d'], friction_normalized['v0'],
            stable] )

inst = Inst(0, 'violin', 3, 'e')
#inst = Inst(0, 'violin', 2, 'a')
for i in range(5):
    inst.new_constants()
    inst.save_consts()

data = inst.attempts

arff.dump('result.arff', data, relation='whatever',
    names=[
        'T', 'L', 'd', 'pl', 'E', 'B1', 'B2',
        's', 'd', 'v0',
        'stable'])

