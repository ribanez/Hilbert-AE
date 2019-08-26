import torch
import torch.utils.data
import h5py


class Dataset_Hilbert(torch.utils.data.Dataset):

    def __init__(self, filename):
        super(Dataset_Hilbert, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_seq = self.h5pyfile['primary'].shape[0]

    def __getitem__(self, index):
        seq = torch.Tensor(self.h5pyfile['primary'][index]).type(dtype=torch.long)

        seq_hilbert = torch.Tensor(self.h5pyfile['hilbert_map'][index, :, :, :]).type(dtype=torch.long)

        return seq, seq_hilbert

    def __len__(self):
        return self.num_seq

    def merge_samples_to_minibatch(samples):
        samples_list = []
        for s in samples:
            samples_list.append(s)
        # sort according to length of aa sequence
        samples_list.sort(key=lambda x: len(x[0]), reverse=True)
        return zip(*samples_list)

    
def DataLoader(filename):
    return Dataset_Hilbert(filename)


def contruct_dataloader_from_disk(filename, minibatch_size):
    return torch.utils.data.DataLoader(DataLoader(filename),
                                       batch_size=minibatch_size,
                                       shuffle=True,
                                       collate_fn=Dataset_Hilbert.merge_samples_to_minibatch)
