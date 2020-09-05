import os
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from os import listdir
from os.path import isfile, split, join
from midi2audio import FluidSynth
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class Preprocess:
    def __init__(self, root, midi_path):
        self.root = root
        self.midi_path = midi_path
        self.wav_path = ''
        self.prefix = 0

    def split_midi(self, mid_file, target_dir, default_tempo=500000, target_segment_len=1):
        song_name = split(mid_file)[-1][:-4]
        mid = MidiFile(mid_file)

        # identify the meta messages
        metas = []
        tempo = default_tempo
        for msg in mid:
            if msg.type is 'set_tempo':
                tempo = msg.tempo
            if msg.is_meta:
                metas.append(msg)
        for meta in metas:
            meta.time = int(mido.second2tick(meta.time, mid.ticks_per_beat, tempo))

        target = MidiFile()
        track = MidiTrack()
        track.extend(metas)
        target.tracks.append(track)

        time_elapsed = 0
        for msg in mid:
            # Skip non-note related messages
            if msg.is_meta:
                continue
            time_elapsed += msg.time
            if msg.type is not 'end_of_track':
                msg.time = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
                track.append(msg)
            if msg.type is 'end_of_track' or time_elapsed >= target_segment_len:
                track.append(MetaMessage('end_of_track'))
                target.save(join(target_dir, '{}.midi'.format(self.prefix)))
                target = MidiFile()
                track = MidiTrack()
                track.extend(metas)
                target.tracks.append(track)
                time_elapsed = 0
                self.prefix += 1

    def midi_to_wav(self, out_path):
        fs = FluidSynth(sample_rate=16000, sound_font='Wii_Grand_Piano.sf2')
        full_path = self.root + '/' + 'midi_split'
        self.wav_path = self.root + '/' + out_path

        for file in os.listdir(full_path):
            if file.endswith('.midi'):
                out_file = self.wav_path + '/' + file.strip('.midi') + '.wav'
                fs.midi_to_audio(full_path + '/' + file, out_file)

    def generate_cqt(self, out_path):
        full_path = self.root + '/wav'

        for file_name in os.listdir(full_path):
            y, sr = librosa.load(full_path + '/' + file_name)
            C = librosa.cqt(y, sr=16000)
            librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                     sr=sr)
            plt.axis('off')
            plt.savefig(out_path + '/' + file_name.strip('.wav'), bbox_inches="tight")
            plt.close('all')





