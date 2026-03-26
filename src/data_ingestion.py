from src.data_preprocessor import preprocess, preprocess_eval
from tensorflow.data import AUTOTUNE


def get_data(train_ds1, test_ds1):
    train_ds = train_ds1.map(preprocess, num_parallel_calls=AUTOTUNE) # Added num_parallel_calls
    train_ds = train_ds.shuffle(2000).cache().prefetch(AUTOTUNE) # Corrected typo train_dstrain_ds to train_ds
    test_ds = test_ds1.map(preprocess_eval, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, test_ds
