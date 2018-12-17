from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 1
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 5

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 20
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

config.data_path = "preprocessed_data2"
config.patient_numbers = [556, 557, 808]
config.n_eval = 1
