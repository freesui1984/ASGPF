# Channels of interest
INCLUDED_CHANNELS = [
    'EEG FP1',
    'EEG FP2',
    'EEG F3',
    'EEG F4',
    'EEG C3',
    'EEG C4',
    'EEG P3',
    'EEG P4',
    'EEG O1',
    'EEG O2',
    'EEG F7',
    'EEG F8',
    'EEG T3',
    'EEG T4',
    'EEG T5',
    'EEG T6',
    'EEG FZ',
    'EEG CZ',
    'EEG PZ']
INCLUDED_CHANNELS_BIDS = [
    'Fp1',
    'Fp2',
    'F3',
    'F4',
    'C3',
    'C4',
    'P3',
    'P4',
    'O1',
    'O2',
    'F7',
    'F8',
    'T3',
    'T4',
    'T5',
    'T6',
    'Fz',
    'Cz',
    'Pz']
# Resampling frequency
# FREQUENCY = 200
FREQUENCY = 200
FREQUENCY_BIDS = 200

# All seizure labels available in TUH
ALL_LABEL_DICT = {'fnsz': 0, 'gnsz': 1, 'spsz': 2, 'cpsz': 3,
                  'absz': 4, 'tnsz': 5, 'tcsz': 6, 'mysz': 7}


# GENERAL CONSTANTS:
VERY_SMALL_NUMBER = 1e-12
INF = 1e20