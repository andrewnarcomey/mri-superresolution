from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

config.data_path = "LowRes Project copy"
# config.patient_numbers = [556, 557, 808, 878, 1012, 1037, 1043, 1133, 1204, 1235, 1252, 1262, 1318, 
#                   1353, 1358, 1386, 1459, 1495, 1518, 1522, 1548, 1608, 1646, 1655, 1665, 
#                   1753, 1769, 1786, 1870, 1973, 2011, 2015]

# config.patient_numbers = [556, 557, 808, 878, 1012, 1037, 1043, 1262, 1318, 1522, 1548, 1608, 1665, 
#                    1753, 2011, 2015]

config.patient_numbers = [556, 557]

## train set location
config.TRAIN.hr_img_path = 'Data/DIV2K_train_HR/'
config.TRAIN.lr_img_path = 'Data/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'Data/DIV2K_valid_HR/'
config.VALID.lr_img_path = 'Data/DIV2K_valid_LR_bicubic/X4/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
