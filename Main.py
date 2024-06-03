import os
import argparse
import numpy as np
from datetime import datetime
from torch import nn, optim
from torch.nn import functional as F
import torch
import Utils, TCN_graphormer
error_if_nonfinite=False
class ModelTrainer(object):
    def __init__(self, params: dict, data: dict, data_container):
        self.params = params
        self.data_container = data_container 
        self.model = self.get_model().to(params['GPU'])
        self.criterion = self.get_loss() 
        self.optimizer = self.get_optimizer() 
        self.clip = 5 
        self.task_level = 1 
        self.cl = True

    def get_model(self):
        model = TCN_graphormer.spatio(num_node = self.params['N'], in_dim=6, blocks=2, layers=3,
                                     in_len=self.params['obs_len'], out_len=self.params['pred_len'], dropout=0.5)    
        return model

    def get_loss(self):
        if self.params['loss'] == 'MSE':
            criterion = nn.MSELoss()
        elif self.params['loss'] == 'MAE':
            criterion = nn.L1Loss()
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.params['learn_rate'], weight_decay=self.params['weight_decay'])
        return optimizer
    
    def train(self, data_loader: dict, modes: list, early_stop_patience=30):
       
        checkpoint = {'epoch': 0, 'state_dict': self.model.state_dict()}
        val_loss = np.inf 
        patience_count = early_stop_patience 

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S')) 
        print(f'     {self.params["model"]} model training begins:')
        #modes=['train','validate','test']
        for epoch in range(1, 1 + self.params['num_epochs']): 
            starttime = datetime.now() 
            running_loss = {mode: 0.0 for mode in modes} #{'train': 0.0, 'validate': 0.0, 'test': 0.0}
            for mode in modes: 
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                step = 0  
                for x_od, x_node, x_SIR, y_S, y_I, y_R, y_Cumulative_confirm,y_new_confirm, attn_bias, edge_input, spatial_pos, Intra_city_travel_intensity in data_loader[mode]:
                    with torch.set_grad_enabled(mode=(mode == 'train')):

                        y_new_confirm_pred, y_S_pred, y_I_pred, y_R_pred, y_Cumulative_confirm_pred, beta, contact ,R0, W_arrive, m_arrive,j0_arrive,N_arrive,X_arrive,Pd_arrive= self.model(attn_bias, spatial_pos, x_node, edge_input, x_SIR, x_od, Intra_city_travel_intensity)
                        if mode == 'train':
                            self.optimizer.zero_grad()
                            if epoch % 2 == 0 and self.task_level <= 14: 
                                self.task_level += 1 
                            if self.cl:
                                loss_new_confirm = self.criterion( y_new_confirm_pred[:, :self.task_level, :, :],y_new_confirm[:, :self.task_level, :, :])
                              
                                loss_Cumulative_confirm = self.criterion(y_Cumulative_confirm_pred[:, :self.task_level, :, :],y_Cumulative_confirm[:, :self.task_level, :, :])

                                loss = loss_new_confirm  + loss_Cumulative_confirm
                                # loss=loss_Cumulative_confirm
                            else:
                                loss_new_confirm = self.criterion( y_new_confirm_pred,y_new_confirm)
                                loss_Cumulative_confirm = self.criterion(y_Cumulative_confirm_pred,y_Cumulative_confirm)

                                loss = loss_new_confirm  + loss_Cumulative_confirm
                                # loss=loss_Cumulative_confirm
                            loss.backward()  
                            if self.clip is not None:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                            self.optimizer.step() 
                        else:

                            loss_new_confirm = self.criterion( y_new_confirm_pred,y_new_confirm)
                            loss_Cumulative_confirm = self.criterion(y_Cumulative_confirm_pred,y_Cumulative_confirm)

                            loss = loss_new_confirm  + loss_Cumulative_confirm
                            # loss=loss_Cumulative_confirm

                    running_loss[mode] += loss * y_Cumulative_confirm.shape[0]  # loss reduction='mean': batchwise average
                    step += y_Cumulative_confirm.shape[0]
                    #torch.cuda.empty_cache()

                # epoch end: evaluate on validation set for early stopping
                if mode == 'validate':
                    epoch_val_loss = running_loss[mode] / step
                    if epoch_val_loss <= val_loss:
                        print(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                              f'Update model checkpoint..', f'used {(datetime.now() - starttime).seconds}s')
                        val_loss = epoch_val_loss
                        torch.save(checkpoint, self.params['output_dir'] + f'/{self.params["model"]}_od.pkl')
                        patience_count = early_stop_patience
                    else:
                        print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.',
                              f'used {(datetime.now() - starttime).seconds}s')
                        patience_count -= 1
                        if patience_count == 0:
                            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                            print(f'    Early stopping at epoch {epoch}. {self.params["model"]} model training ends.')
                            return
        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training ends.') #
        return

    def test(self, data_loader: dict, modes: list):
        trained_checkpoint = torch.load(self.params['output_dir'] + f'/{self.params["model"]}_od.pkl')
        self.model.load_state_dict(trained_checkpoint['state_dict'])
        self.model.eval()

        for mode in modes:
            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            print(f'     {self.params["model"]} model testing on {mode} data begins:')
           
            forecast_Cumulative_confirm, ground_truth_Cumulative_confirm = [], []
            forecast_new_confirm, ground_truth_new_confirm = [], []
            cond1,cond2 =[], []
            R0_forecase = []
            forecast_W_arrive=[]
            forecast_m_arrive=[]    
            forecast_j0_arrive=[]
            forecast_N_arrive=[]
            forecast_X_arrive=[]
            forecast_Pd_arrive=[]

            for x_od, x_node, x_SIR, y_S, y_I, y_R, y_Cumulative_confirm, y_new_confirm, attn_bias, edge_input, spatial_pos, Intra_city_travel_intensity in data_loader[mode]:
                
                y_new_confirm_pred, y_S_pred, y_I_pred, y_R_pred, y_Cumulative_confirm_pred, beta, contact,R0,W_arrive, m_arrive,j0_arrive,N_arrive,X_arrive,Pd_arrive = self.model(attn_bias, spatial_pos, x_node, edge_input, x_SIR, x_od, Intra_city_travel_intensity)
        
                # forecast_S.append(y_S_pred.cpu().detach())
                # ground_truth_S.append(y_S.cpu().detach())
                # forecast_I.append(y_I_pred.cpu().detach())
                # ground_truth_I.append(y_I.cpu().detach())
                # forecast_R.append(y_R_pred.cpu().detach())
                # ground_truth_R.append(y_R.cpu().detach())
                forecast_Cumulative_confirm.append(y_Cumulative_confirm_pred.cpu().detach())
                ground_truth_Cumulative_confirm.append(y_Cumulative_confirm.cpu().detach())
                forecast_new_confirm.append(y_new_confirm_pred.cpu().detach())
                ground_truth_new_confirm.append(y_new_confirm.cpu().detach())
                cond1.append(beta.cpu().detach())
                cond2.append(contact.cpu().detach())
                R0_forecase.append(R0.cpu().detach())

                forecast_W_arrive.append(W_arrive.cpu().detach())
                forecast_m_arrive.append(m_arrive.cpu().detach())
                forecast_j0_arrive.append(j0_arrive.cpu().detach())
                forecast_N_arrive.append(N_arrive.cpu().detach())
                forecast_X_arrive.append(X_arrive.cpu().detach())
                forecast_Pd_arrive.append(Pd_arrive.cpu().detach())

            forecast_Cumulative_confirm = torch.cat(forecast_Cumulative_confirm, dim=0)
            ground_truth_Cumulative_confirm = torch.cat(ground_truth_Cumulative_confirm, dim=0)
            forecast_new_confirm = torch.cat(forecast_new_confirm, dim=0)
            ground_truth_new_confirm = torch.cat(ground_truth_new_confirm, dim=0)
            cond1=np.concatenate(cond1, 0)
            cond2=np.concatenate(cond2, 0)
            R0_forecase = torch.cat (R0_forecase, dim=0)

            forecast_W_arrive = torch.cat(forecast_W_arrive, dim=0)
            forecast_m_arrive = torch.cat(forecast_m_arrive, dim=0)
            forecast_j0_arrive = torch.cat(forecast_j0_arrive, dim=0)
            forecast_N_arrive = torch.cat(forecast_N_arrive, dim=0)
            forecast_X_arrive = torch.cat(forecast_X_arrive, dim=0)
            forecast_Pd_arrive = torch.cat(forecast_Pd_arrive, dim=0)

            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode + 'Cumulative_confirm_prediction.npy', forecast_Cumulative_confirm)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode + 'Cumulative_confirm_groundtruth.npy', ground_truth_Cumulative_confirm)
            
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode + '_prediction.npy', forecast_new_confirm)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode + '_groundtruth.npy', ground_truth_new_confirm)
            
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode +  'para_b.npy', cond1)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode +  'contact.npy', cond2)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode +  'R0.npy', R0_forecase)

            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode +  'W_arrive.npy', forecast_W_arrive)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode +  'm_arrive.npy', forecast_m_arrive)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode +  'j0_arrive.npy', forecast_j0_arrive)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode +  'N_arrive.npy', forecast_N_arrive)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode +  'X_arrive.npy', forecast_X_arrive)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode +  'Pd_arrive.npy', forecast_Pd_arrive)
        

            # evaluate on metrics
            MSE, RMSE, MAE, MAPE, RAE = self.evaluate(forecast_Cumulative_confirm + forecast_new_confirm ,ground_truth_Cumulative_confirm + ground_truth_new_confirm)
            f = open(self.params['output_dir'] + '/' + self.params['model'] + '_prediction_scores.txt', 'a')
            f.write("%s, MSE, RMSE, MAE, MAPE, RAE, %.10f, %.10f, %.10f, %.10f, %.10f\n" % (mode, MSE, RMSE, MAE, MAPE, RAE))
            if mode == 'test':
                for i in range(ground_truth_Cumulative_confirm.shape[1]):
                    print("%d step" % (i+1))
                    MSE, RMSE, MAE, MAPE, RAE = self.evaluate(forecast_Cumulative_confirm[:, i, :] + forecast_new_confirm[:, i, :], ground_truth_Cumulative_confirm[:, i, :] + ground_truth_new_confirm[:, i, :])
                    f.write("%d step,  %s, MSE, RMSE, MAE, MAPE, RAE, %.10f, %.10f, %.10f, %.10f, %.10f\n" % (
                        i + 1, mode, MSE, RMSE, MAE, MAPE, RAE))
            f.close()

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model testing ends.')
        
        return

    @staticmethod
    def evaluate(y_pred: np.array, y_true: np.array):
        def MSE(y_pred: np.array, y_true: np.array):  
            return F.mse_loss(y_pred, y_true)

        def RMSE(y_pred: np.array, y_true: np.array): 
            return torch.sqrt(F.mse_loss(y_pred, y_true))

        def MAE(y_pred: np.array, y_true: np.array): 
            return F.l1_loss(y_pred, y_true)

        def MAPE(y_pred: np.array, y_true: np.array): 
            return mape(y_pred, y_true + 1.0)

        def RAE(y_pred: np.array, y_true: np.array):
            return rae(y_pred, y_true)

        print('MSE:', round(MSE(y_pred, y_true).item(), 4))
        print('RMSE:', round(RMSE(y_pred, y_true).item(), 4))
        print('MAE:', round(MAE(y_pred, y_true).item(), 4))
        print('MAPE:', round((MAPE(y_pred, y_true).item() * 100), 4), '%')
        print('RAE:', round(RAE(y_pred, y_true).item(), 4))
        return MSE(y_pred, y_true), RMSE(y_pred, y_true), MAE(y_pred, y_true), MAPE(y_pred, y_true), RAE(y_pred, y_true)

def mape(preds, labels):
    loss = torch.abs(preds-labels)/labels
    return torch.mean(loss)

def rae(preds, labels):
    loss = torch.abs(preds-labels)/(torch.abs(labels-labels.mean()).sum())
    return torch.sum(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Prediction')

    # command line arguments
    parser.add_argument('-GPU', '--GPU', type=str, help='Specify GPU usage', default='cuda:0')
    parser.add_argument('-in', '--input_dir', type=str, default='../data') 
    parser.add_argument('-out', '--output_dir', type=str, default='./output') 
    parser.add_argument('-model', '--model', type=str, help='Specify model', choices=['TCN_graphormer'], default='TCN_graphormer')
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation sequence', default=3) 
    parser.add_argument('-pred', '--pred_len', type=int, help='Length of prediction sequence', default=3) 
    parser.add_argument('-split', '--split_ratio', type=float, nargs='+',
                        help='Relative data split ratio in train : validate : test'
                             ' Example: -split 6 1 1', default=[3, 1, 1]) 
    parser.add_argument('-batch', '--batch_size', type=int, default=32) 
    parser.add_argument('-epoch', '--num_epochs', type=int, default=3000)
    parser.add_argument('-loss', '--loss', type=str, help='Specify loss function',
                        choices=['MSE', 'MAE'], default='MAE')  
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3) 
    parser.add_argument('-test', '--test_only', type=int, default=0) 
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6)
    parser.add_argument('-rep', '--reproduce', type=bool, default=True)
    parser.add_argument('-rd', '--random_seed', type=int, default=3)

    params = parser.parse_args().__dict__  

    if params['reproduce'] is True:
       
        torch.manual_seed(params['random_seed'])
        torch.cuda.manual_seed(params['random_seed'])
        np.random.seed(params['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(params['output_dir'], exist_ok=True)
    # torch.backends.cudnn.enabled = False
    
    data_input = Utils.DataInput(data_dir=params['input_dir'], data_split_ratio=params['split_ratio']) 
    data = data_input.load_data()   # dataset
    params['N'] = data['node'].shape[1]

  
    data_generator = Utils.DataGenerator(obs_len=params['obs_len'],
                                         pred_len=params['pred_len'],
                                         data_split_ratio=params['split_ratio'])
    data_loader = data_generator.get_data_loader(data=data, params=params)  #data_loader


    trainer = ModelTrainer(params=params, data=data, data_container=data_input)

    if bool(params['test_only']) == False:
        trainer.train(data_loader=data_loader,
                      modes=['train', 'validate'])
    trainer.test(data_loader=data_loader,
                 modes=['train', 'validate', 'test'])

