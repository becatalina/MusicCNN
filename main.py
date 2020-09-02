from preprocess import Preprocess


def main():
    proc = Preprocess('Dataset', 'music')
    proc.midi_to_wav('wav')


main()
