import json
import pandas as pd



import random
from pytorchvideo.transforms import RandAugment
from torchvision import transforms
from torch.utils import data



from dataset.ucfhmdb_dataset import UCFHMDBDataset
from dataset.epic_dataset import EpicKitchenDataset
from dataset import transforms as T
from torch.utils.data import DataLoader
from utils.utils import seed_worker

def get_data(transform, args, mode, dataset, pseudo_label_path = None):

    args.dataset = dataset
    args.load_type = mode
    
    if pseudo_label_path is not None:
        if args.target_dataset in ["D1", "D2", "D3"]:
            pseudo_labels = pd.read_pickle(pseudo_label_path)
        else:
            pseudo_labels = json.load(open(pseudo_label_path, 'rb'))
        print("Loading pseudo-labels from {}".format(pseudo_label_path))
    else:
        pseudo_labels = None
        
    
    if args.dataset in ['ucf101', 'hmdb51', 'ucf101s', 'hmdb51s']:
        dataset = UCFHMDBDataset(args, transform = transform, pseudo_labels = pseudo_labels)
    else:
        dataset = EpicKitchenDataset(args, transform = transform, pseudo_labels = pseudo_labels)

    return dataset



def get_dataloader(args, mode, dataset):
    
    args.load_type = mode
    
    if mode == 'train':
        if args.modality == "Joint":
            batch_size = args.batch_size // 2
        else:
            batch_size = args.batch_size 
        data_loader = DataLoader(
                dataset, batch_size = batch_size , shuffle = True,
                num_workers = args.num_workers, worker_init_fn=seed_worker,
                pin_memory = False, drop_last = False
            )

    elif mode == 'val':
        if args.modality == "Joint":
            batch_size = args.batch_size // 2
        else:
            batch_size = args.batch_size 
        data_loader = DataLoader(
            dataset, batch_size = batch_size, shuffle = False,
            worker_init_fn=seed_worker,
            num_workers = args.num_workers, pin_memory = False,
            drop_last = False
        )

    elif mode == "generate-pseudo-label" or mode == "feature":
        
        data_loader = DataLoader(
            dataset, batch_size = 1, shuffle = False,
            num_workers = args.num_workers, pin_memory = False,
            drop_last = False
        )
    
    print("Mode: {} Size: {}".format(mode, len(dataset)))

    return data_loader

    


def get_weak_transforms(args, mode):

    if mode == 'train':
        transform = transforms.Compose([T.RandomCrop(224),
        T.RandomHorizontalFlip()
        ])
    else:
        transform = transforms.Compose([T.CenterCrop(224)
        ])

    return transform


def get_strong_transforms(args, mode):

    transform = transforms.Compose([
                                    RandAugment(
                                    magnitude = 9,
                                    num_layers = 3,
                                    prob = 0.5
                                    ),
                                    T.change_time_dimension(),
                                    T.CenterCrop(224)
                                    ]
                                )
    
    return transform



# class GaussianBlur(object):
#     def __init__(self, sigma=[0.1, 2.0]):
#         self.sigma = sigma

#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x


# # returns dataset mean and std
# def get_mean_std():
#     mean = [0.4345, 0.4051, 0.3775]
#     std = [0.2768, 0.2713, 0.2737]
#     return mean, std

# def get_weak_transforms(args, mode):
#     if mode == 'train':
#         transformations = []
#         # resize
#         transformations = [(TF.resize, (256, 256))]
#         # random crop
#         dummy = Image.new("RGB", (256, 256))
#         i, j, h, w = transforms.RandomCrop.get_params(dummy, (224, 224))
#         transformations.append((TF.resized_crop, i, j, h, w, (224, 224)))
        
#         if random.random() < 0.5:
#             transformations.append((TF.hflip,))
            
#         # normalization
#         mean, std = get_mean_std()
#         transformations.append((TF.normalize, mean, std))
        
#     else:
#         transformations = []
#         transformations.append((TF.center_crop, (224, 224)))
#         # normalization
#         mean, std = get_mean_std()
#         transformations.append((TF.normalize, mean, std))
    
#     return transformations

# def get_strong_transforms(args, mode):
#     if mode == 'train':
#         transformations = []
#         # resize
#         transformations = [(TF.resize, (256, 256))]
#         # random crop
#         dummy = Image.new("RGB", (256, 256))
#         i, j, h, w = transforms.RandomCrop.get_params(dummy, (224, 224))
#         transformations.append((TF.resized_crop, i, j, h, w, (224, 224)))
        
#         if random.random() < 0.5:
#             transformations.append((TF.hflip,))
            
#         # normalization
#         mean, std = get_mean_std()
#         transformations.append((TF.normalize, mean, std))
        
#     else:
#         transformations = []
#         transformations.append((TF.center_crop, (224, 224)))
#         # normalization
#         mean, std = get_mean_std()
#         transformations.append((TF.normalize, mean, std))
    
#     return transformations