import random

def split_train_val_index(tuples, aug_factor, piece_size = 1, val_ratio = 0.2):
    train_index = []
    val_index = []
    tuples_num = len(tuples) // aug_factor
    piece_num = tuples_num // piece_size
    if int(piece_num * val_ratio) < 1:
        print("Can't fit validation ratio, check piece size or validation ratio")
    val_piece_index = random.sample(range(piece_num), max(int(piece_num * val_ratio),1))
    for i in range(len(tuples)):
        if (i // aug_factor) // piece_size in val_piece_index:
            val_index.append(i)
        else:
            train_index.append(i)
    return train_index, val_index

tuples_list = [(i, ) for i in range(1000)]
train_idx, val_idx = split_train_val_index(tuples_list, 5, 3, 0.2)
print(len(train_idx), train_idx)
print(len(val_idx), val_idx)
