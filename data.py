import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import json
from models import Custom3DCNN, PatchEmbeddings
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

class MultiModalDataset(Dataset):
    def __init__(self, data_dict, observed_idx, ids, labels, input_dims, transforms, masks, use_common_ids=True):
        self.data_dict = data_dict
        self.mc = np.array(data_dict['modality_comb'])
        self.observed = observed_idx
        self.ids = ids
        self.labels = labels
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.use_common_ids = use_common_ids
        self.data_new = {modality: data[ids] for modality, data in self.data_dict.items() if 'modality' not in modality}
        self.label_new = self.labels[ids]
        self.mc_new = self.mc[ids]
        self.observed_new = self.observed[ids]

        # Sort ids by the number of available modalities
        self.sorted_ids = sorted(np.arange(len(ids)), key=lambda idx: sum([1 for modality in self.data_new if -2 not in self.data_new[modality][idx]]), reverse=True)
        self.data_new = {modality: data[self.sorted_ids] for modality, data in self.data_new.items()}
        self.label_new = self.label_new[self.sorted_ids]
        self.mc_new = self.mc_new[self.sorted_ids]
        self.observed_new = self.observed_new[self.sorted_ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_data = {}
        for modality, data in self.data_new.items():
            sample_data[modality] = data[idx]
            if modality == 'image':
                subj1 = data[idx]
                subj_gm_3d = np.zeros(self.masks.shape, dtype=np.float32)
                subj_gm_3d.ravel()[self.masks] = subj1
                subj_gm_3d = subj_gm_3d.reshape((91, 109, 91))
                if self.transforms:
                    subj_gm_3d = self.transforms(subj_gm_3d)
                sample = subj_gm_3d[None, :, :, :]  # Add channel dimension
                sample_data[modality] = np.array(sample)

        label = self.label_new[idx]
        mc = self.mc_new[idx]
        observed = self.observed_new[idx]

        return sample_data, label, mc, observed

def convert_ids_to_index(ids, index_map):
    return [index_map[id] if id in index_map else -1 for id in ids]

def load_and_preprocess_image_data(image_path, label_df, id_to_idx):
    # Load and preprocess image data
    image_data = np.load(os.path.join(image_path, 'ADNI_G.npy'), mmap_mode='r')
    mask_path = os.path.join(image_path, 'BLSA_SPGR+MPRAGE_averagetemplate_muse_seg_DS222.nii.gz')
    
    subject_ids = []
    dates = []
    with open('./data/adni/image/ADNI_subj.txt', 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split('_')
            subject_id = '_'.join(parts[:3])
            date = parts[-1]
            subject_ids.append(subject_id)
            dates.append(date)

    df = pd.DataFrame({
            'PTID': subject_ids,
            'date': dates
        })

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date', ascending=False)
    idx = df.groupby('PTID')['date'].idxmax()

    # Creating the subset DataFrame using the indexes
    subdf = df.loc[idx]
    subdf = subdf.sort_index()
    subdf = subdf.reset_index()

    merged_df = pd.merge(subdf, label_df, on='PTID', how='left')

    image_data = image_data[merged_df['index']]
    final_subject_ids = list(subdf.PTID)

    new_idx = np.array(convert_ids_to_index(final_subject_ids, id_to_idx))
    filtered_idx = [x for x in new_idx if x != -1]
    tmp = np.zeros((len(id_to_idx), image_data.shape[1])) - 2
    tmp[filtered_idx] = image_data[np.array(new_idx) != -1]

    data = nib.load(mask_path).get_fdata()
    mean = image_data.mean()
    std = image_data.std()     
    # mean = data.mean()
    # std = data.std()
    mask_gm = (data == 150).ravel()
    
    return tmp, filtered_idx, mean, std, mask_gm


def load_and_preprocess_data(args, modality_dict):
    # Paths
    image_path = './data/adni/image'
    genomic_path = './data/adni/genomic/genomic_merged.h5ad'
    clinical_path = './data/adni/clinical/clinical_merged'
    biospecimen_path = './data/adni/biospecimen/biospecimen_merged'
    label_df = pd.read_csv('./data/adni/label.csv', index_col='PTID')
    label_df['DIAGNOSIS'] -= 1
    labels = label_df['DIAGNOSIS'].values.astype(np.int64)
    n_labels = len(set(labels))

    with open('./data/adni/PTID_splits.json') as json_file:
        data_split = json.load(json_file)

    train_ids = list(set(data_split['training']))
    valid_ids = list(set(data_split['validation']))
    test_ids = list(set(data_split['testing']))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0],4), dtype=bool) # IGCB order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if 'I' in args.modality or 'i' in args.modality:
        arr, filtered_idx, mean, std, mask = load_and_preprocess_image_data(image_path, label_df, id_to_idx)
        observed_idx_arr[:, modality_dict['image']] = arr[:, 0] != -2
        for idx in filtered_idx:
            update_modality_combinations(idx, 'I')

        data_dict['image'] = np.array(arr)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['image'] = torch.nn.Sequential(
            Custom3DCNN(hidden_dim=args.hidden_dim).to(args.device),
            PatchEmbeddings(feature_size=args.hidden_dim, num_patches=args.num_patches, embed_dim=args.hidden_dim).to(args.device)
            )
        input_dims['image'] = arr.shape[1]
        transforms['image'] = Compose([
                                    ToTensor(),
                                    Normalize(mean=[mean], std=[std]),
                                ])
        masks['image'] = mask

    if 'G' in args.modality or 'g' in args.modality:
        df = sc.read_h5ad(genomic_path).to_df()
        if args.initial_filling == 'mean':
            df = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=0) # use mode as genotype values are 0,1,2
        arr = df.values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        arr = scaler.fit_transform(arr)
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['genomic']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'G')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict['genomic'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['genomic'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['genomic'] = df.shape[1]

    if 'C' in args.modality or 'c' in args.modality:
        if args.initial_filling == 'mean':
            path = clinical_path + '_mean.csv'
        else:
            path = clinical_path + '.csv'
        df = pd.read_csv(path, index_col=0)
        columns_to_exclude = [col for col in df.columns if col.startswith('PTCOGBEG') or col.startswith('PTADDX') or col.startswith('PTADBEG')]
        if len(columns_to_exclude) > 0:
            df = df.drop(columns_to_exclude, axis=1)
        arr = df.values.astype(np.float32)
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['clinical']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'C')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['clinical'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['clinical'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['clinical'] = df.shape[1]

    if 'B' in args.modality or 'b' in args.modality:
        if args.initial_filling == 'mean':
            path = biospecimen_path + '_mean.csv'
        else:
            path = biospecimen_path + '.csv'
        df = pd.read_csv(path, index_col=0)
        arr = df.values
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['biospecimen']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'B')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['biospecimen'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['biospecimen'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['biospecimen'] = df.shape[1]

    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0) # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index

def load_and_preprocess_data_mimic(args, modality_dict):
    # Paths
    lab_path = './data/mimic/lab_x'
    note_path = './data/mimic/note_x'
    code_path = './data/mimic/code_x'
    label_df = pd.read_csv('./data/mimic/labels.csv', index_col='subject_id')
    labels = label_df['one_year_mortality'].values.astype(np.int64)
    n_labels = len(set(labels))

    with open('./data/mimic/PTID_splits_mimic.json') as json_file:
        data_split = json.load(json_file)

    train_ids = list(set(data_split['training']))
    valid_ids = list(set(data_split['validation']))
    test_ids = list(set(data_split['testing']))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0], args.n_full_modalities), dtype=bool) # IGCB order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if 'L' in args.modality or 'l' in args.modality:
        path = lab_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['lab']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'L')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['lab'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['lab'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['lab'] = arr.shape[1]

    if 'N' in args.modality or 'n' in args.modality:
        path = note_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['note']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'N')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['note'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['note'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['note'] = arr.shape[1]

    if 'C' in args.modality or 'c' in args.modality:
        path = code_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['code']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'C')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['code'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['code'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['code'] = arr.shape[1]
    
    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0) # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index

def collate_fn(batch):
    data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data]), dtype=torch.float32) for modality in modalities}
    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, labels, mcs, observeds

def create_loaders(data_dict, observed_idx, labels, train_ids, valid_ids, test_ids, batch_size, num_workers, pin_memory, input_dims, transforms, masks, use_common_ids=True):
    if 'image' in list(data_dict.keys()):
        train_transfrom = val_transform = test_transform = transforms['image']
        # val_transform = test_transform = False
        mask = masks['image']
    else:
        train_transfrom = val_transform = test_transform = False
        mask = None

    train_dataset = MultiModalDataset(data_dict, observed_idx, train_ids, labels, input_dims, train_transfrom, mask, use_common_ids)
    valid_dataset = MultiModalDataset(data_dict, observed_idx, valid_ids, labels, input_dims, val_transform, mask, use_common_ids)
    test_dataset = MultiModalDataset(data_dict, observed_idx, test_ids, labels, input_dims, test_transform, mask, use_common_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, train_loader_shuffle, val_loader, test_loader

# Updated: full modality index is 0.
def get_modality_combinations(modalities):
    all_combinations = []
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)
    
    # Create a mapping dictionary
    combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
    return combination_to_index