import numpy as np


def hilbert_curve(n):
    if n == 1:
        return np.zeros((1, 1), np.int32)

    t = hilbert_curve(n // 2)

    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size * 2
    d = np.flipud(np.rot90(t, -1)) + t.size * 3

    return np.vstack(map(np.hstack, [[a, b], [d, c]]))


def seq2hilbert(sequence, hilbert_map, channels=20):

    map_ = np.zeros([hilbert_map.shape[0], hilbert_map.shape[1], channels])

    for idx, row in enumerate(hilbert_map):

        for jdx, ii in enumerate(row):

            if len(sequence) <= ii:

                continue

            aa = sequence[ii]

            if aa > channels:
                raise Exception("index > number of channels")

            map_[idx, jdx, aa] = 1

    return map_


if __name__ == '__main__':

    # LITTLE TEST !
    # HILBERT - MAP
    #  0,  3,  4,  5
    #  1,  2,  7,  6
    # 14, 13,  8,  9
    # 15, 12, 11, 10
    #                   0 1 2 3 4 5 6 7 8 9 10 11

    order = 4

    hilbert_map = hilbert_curve(order)

    map_ = seq2hilbert([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], hilbert_map, 2)

    true_map = np.array([[[1, 0], [1, 0], [0, 1], [0, 1]],
                         [[1, 0], [1, 0], [0, 1], [0, 1]],
                         [[0, 0], [0, 0], [1, 0], [1, 0]],
                         [[0, 0], [0, 0], [1, 0], [1, 0]]])

    assert (map_ == true_map).all()
