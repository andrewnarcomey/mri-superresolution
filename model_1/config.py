from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 1
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 1

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 5
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

config.data_path = "preprocessed_data"
config.patient_numbers = [556, 557, 808, 878, 1012, 1037, 1043, 1262, 1318, 1522, 1548, 1608, 1665, # patients with 512x512 dimensions
                  1753, 2011, 2015]
config.n_eval = 5
