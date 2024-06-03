import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DataInput(object):
    def __init__(self, data_dir: str, data_split_ratio: tuple):
        self.data_dir = data_dir
        self.data_split_ratio = data_split_ratio

    def load_data(self):
        
        data = np.load('total_china.npy', allow_pickle='TRUE').item() 
        attn_bias = np.load('attn_bias_china.npy')
        in_degree = np.load('in_degree_china.npy')
        out_degree = np.load('out_degree_china.npy')
        edge_input = np.load('edge_input_china.npy')
        spatial_pos = np.load('spd_china.npy')
        risk_rate = np.load('risk_rate_china.npy')
        Cumulative_confirm = np.load('Cumulative_confirm_china.npy')
        Intra_city_travel_intensity = np.load('Intra_city_travel_intensity_china.npy')
        

        data_od = data['od'] 
        data_node_inf = np.log(data['node'][...,[0]]+1.0) 
        # data_node_inf = np.log(Cumulative_confirm)
        data_node_other = data['node'][...,[1,2]]
        data_node = np.concatenate((data_node_inf,data_node_other,in_degree,out_degree,risk_rate),axis=-1) 
        
        data_SIR = np.concatenate((data['SIR'],Cumulative_confirm),axis=-1).astype('float')  

        data_y_S = data['SIR'][...,[0]] 
        data_y_I = data['SIR'][...,[1]] 
        data_y_R = data['SIR'][...,[2]] 
        data_y_Cumulative_confirm = Cumulative_confirm
        data_y_new_confirm = data['node'][...,[0]] 

        dataset = dict()
        dataset['od'] = data_od
        dataset['node'] = data_node
        dataset['SIR'] = data_SIR
        
        dataset['attn_bias'] = attn_bias
        dataset['edge_input'] = edge_input
        dataset['spatial_pos'] = spatial_pos

        dataset['y_S'] = data_y_S
        dataset['y_I'] = data_y_I
        dataset['y_R'] = data_y_R
        dataset['y_Cumulative_confirm'] = data_y_Cumulative_confirm
        dataset['y_new_confirm'] = data_y_new_confirm
        dataset['Intra_city_travel_intensity'] = Intra_city_travel_intensity
        return dataset


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class ODDataset(Dataset):
    def __init__(self, inputs: dict, output: dict, mode: str, mode_len: dict):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_od'][item], self.inputs['x_node'][item], self.inputs['x_SIR'][item], self.output['y_S'][item],  self.output['y_I'][item],  self.output['y_R'][item],  self.output['y_Cumulative_confirm'][item], self.output['y_new_confirm'][item],self.inputs['attn_bias'][item],self.inputs['edge_input'][item],self.inputs['spatial_pos'][item],self.inputs['Intra_city_travel_intensity'][item]

    def prepare_xy(self, inputs: dict, output: dict):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:  # test
            start_idx = self.mode_len['train'] + self.mode_len['validate']

        x = dict()
        x['x_od'] = inputs['x_od'][start_idx: (start_idx + self.mode_len[self.mode])]
        x['x_SIR'] = inputs['x_SIR'][start_idx: (start_idx + self.mode_len[self.mode])]
        x['x_node'] = inputs['x_node'][start_idx: (start_idx + self.mode_len[self.mode])]

        x['attn_bias'] = inputs['attn_bias'][start_idx: (start_idx + self.mode_len[self.mode])]
        x['edge_input'] = inputs['edge_input'][start_idx: (start_idx + self.mode_len[self.mode])]
        x['spatial_pos'] = inputs['spatial_pos'][start_idx: (start_idx + self.mode_len[self.mode])]
        x['Intra_city_travel_intensity'] = inputs['Intra_city_travel_intensity'][start_idx: (start_idx + self.mode_len[self.mode])]

        y=dict()
        y['y_S'] = output['y_S'][start_idx: start_idx + self.mode_len[self.mode]]
        y['y_I'] = output['y_I'][start_idx: start_idx + self.mode_len[self.mode]]
        y['y_R'] = output['y_R'][start_idx: start_idx + self.mode_len[self.mode]]
        y['y_Cumulative_confirm'] = output['y_Cumulative_confirm'][start_idx: start_idx + self.mode_len[self.mode]]
        y['y_new_confirm'] = output['y_new_confirm'][start_idx: start_idx + self.mode_len[self.mode]]

        return x, y

class DataGenerator(object):
    def __init__(self, obs_len: int, pred_len, data_split_ratio: tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len: int):
        mode_len = dict()
        mode_len['train'] = int(self.data_split_ratio[0] / sum(self.data_split_ratio) * data_len)
        mode_len['validate'] = int(self.data_split_ratio[1] / sum(self.data_split_ratio) * data_len)
        mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
        return mode_len

    def get_data_loader(self, data: dict, params: dict):

        x_od, x_node, x_SIR, y_S, y_I, y_R, y_Cumulative_confirm ,y_new_confirm, x_attn_bias, x_edge_input, x_spatial_pos, Intra_city_travel_intensity = self.get_feats(data)
        x_od = np.asarray(x_od)
        x_node = np.asarray(x_node)
        x_SIR = np.asarray(x_SIR)

        y_S = np.asarray(y_S)
        y_I = np.asarray(y_I)
        y_R = np.asarray(y_R)
        y_Cumulative_confirm = np.asarray(y_Cumulative_confirm)
        y_new_confirm = np.asarray(y_new_confirm)

        x_attn_bias = np.asarray(x_attn_bias)
        x_edge_input = np.asarray(x_edge_input)
        x_spatial_pos = np.asarray(x_spatial_pos)
        Intra_city_travel_intensity = np.asarray(Intra_city_travel_intensity)

        mode_len = self.split2len(data_len=y_I.shape[0])


#         for i in range(x_node.shape[-1]):
#                 scaler = StandardScaler(mean=x_node[:mode_len['train'],..., i].mean(),
#                                         std=x_node[:mode_len['train'],..., i].std())
#                 x_node[...,i] = scaler.transform(x_node[...,i])

        feat_dict = dict()
        y =dict()
        feat_dict['x_od'] = torch.from_numpy(x_od).float().to(params['GPU'])
        feat_dict['x_node'] = torch.from_numpy(x_node).float().to(params['GPU'])
        feat_dict['x_SIR'] = torch.from_numpy(x_SIR).float().to(params['GPU'])

        y['y_S'] = torch.from_numpy(y_S).float().to(params['GPU'])
        y['y_I'] = torch.from_numpy(y_I).float().to(params['GPU'])
        y['y_R'] = torch.from_numpy(y_R).float().to(params['GPU'])
        y['y_Cumulative_confirm'] = torch.from_numpy(y_Cumulative_confirm).float().to(params['GPU'])
        y['y_new_confirm'] = torch.from_numpy(y_new_confirm).float().to(params['GPU'])

        feat_dict['attn_bias'] = torch.from_numpy(x_attn_bias).float().to(params['GPU'])
        feat_dict['edge_input'] = torch.from_numpy(x_edge_input).float().to(params['GPU'])
        feat_dict['spatial_pos'] = torch.from_numpy(x_spatial_pos).float().to(params['GPU'])
        feat_dict['Intra_city_travel_intensity'] = torch.from_numpy(Intra_city_travel_intensity).float().to(params['GPU'])

        print('Data split:', mode_len)

        data_loader = dict()  # data_loader for [train, validate, test]
        for mode in ['train', 'validate', 'test']:
            dataset = ODDataset(inputs=feat_dict, output=y, mode=mode, mode_len=mode_len)
            print('Data loader', '|', mode, '|', 'input node features:', dataset.inputs['x_node'].shape, '|'
                  'output:', dataset.output['y_Cumulative_confirm'].shape)
            if mode == 'train':
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True)
            else:
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False)
        return data_loader

    def get_feats(self, data: dict):
        x_od, x_node, x_SIR, y_S, y_I, y_R, y_Cumulative_confirm, y_new_confirm , x_attn_bias, x_edge_input, x_spatial_pos, Intra_city_travel_intensity= [], [], [], [], [], [], [], [], [], [], [], []

        for i in range(self.obs_len, data['od'].shape[0] - self.pred_len + 1):
            x_od.append(data['od'][i - self.obs_len: i])
            x_node.append(data['node'][i - self.obs_len: i])
            x_SIR.append(data['SIR'][i - self.obs_len: i])

            y_S.append(data['y_S'][i: i + self.pred_len])
            y_I.append(data['y_I'][i: i + self.pred_len])
            y_R.append(data['y_R'][i: i + self.pred_len])
            y_Cumulative_confirm.append(data['y_Cumulative_confirm'][i: i + self.pred_len])
            y_new_confirm.append(data['y_new_confirm'][i: i + self.pred_len])

            x_attn_bias.append(data['attn_bias'][i - self.obs_len: i])
            x_edge_input.append(data['edge_input'][i - self.obs_len: i])
            x_spatial_pos.append(data['spatial_pos'][i - self.obs_len: i])
            Intra_city_travel_intensity.append(data['Intra_city_travel_intensity'][i - self.obs_len: i])



        return x_od, x_node, x_SIR, y_S, y_I, y_R, y_Cumulative_confirm, y_new_confirm,x_attn_bias, x_edge_input, x_spatial_pos, Intra_city_travel_intensity

