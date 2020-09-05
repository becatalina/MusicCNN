from preprocess import Preprocess
import utils
from os import listdir
import pretty_midi
from numpy import save, load


def create_files():
    proc = Preprocess('Dataset', 'music')
    for file_name in listdir('Dataset/music'):
        proc.split_midi(mid_file='Dataset/music/' + file_name, target_dir='Dataset/midi_split')

    proc.midi_to_wav('wav')
    proc.generate_cqt(out_path='Dataset/cqt')


def generate_vectors():
    for file_name in listdir('Dataset/midi_split'):
        midi_data = pretty_midi.PrettyMIDI('Dataset/midi_split/' + file_name)
        one_hot = utils.pretty_midi_to_one_hot(pm=midi_data, fs=100)
        save(file='Dataset/one_hots/' + file_name.strip('.midi'), arr=one_hot)


def print_matrices():
    p = load('Dataset/one_hots/2.npy')
    print(p)


def main():
    # create_files()
    #generate_vectors()
    print_matrices()
    pass


main()
