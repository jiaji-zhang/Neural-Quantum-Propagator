import os, datetime
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import yaml 
import torch

from train import Data_trainer, Phys_trainer, Training_model, Testing_model, Save_list_dict_excel

from NQP_setups import Setup_logs, Setup_device, Setup_dataloader
from NQP_setups import Setup_FNO, Setup_state_dict, Setup_random_initials
from NQP_setups import Setup_qme, Setup_data_trainer, Setup_phys_trainer
from NQP_setups import Loss_data, Transform_timewise


# NQP: neural quantum propagator
def Main_training(config: yaml.Loader):
     # device
     device = Setup_device()
     # root, input, output dir
     input_dir = config['dir']['root_dir'] + config['dir']['input_dir'] 
     output_dir = config['dir']['root_dir'] + config['dir']['output_dir'] 
     os.makedirs(output_dir, exist_ok=True)
     # wandb log, save checkpoints     
     log_run, save_ckpt = Setup_logs(output_dir)
     # phys_obj, determining the equation type
     obj_phys = Setup_qme(config['phys'], device, input_dir)   
     # create model, optimizers     
     model = Setup_FNO(config['model'], device)    
     # optimizer
     optimizer = torch.optim.Adam(model.parameters(), config['train']['lr_base'])
     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config['train']['milestones'], 
                                                      config['train']['gamma'])  
     # data trainer
     print('Train with dataset and random initials... \n')   
     data_process, data_loss, phys_loss_data = Setup_data_trainer(obj_phys)  
     data_train_obj = Data_trainer(Setup_dataloader(config['train_data'], input_dir, obj_phys),
                                   data_process.Input_preprocess, data_process.Output_postprocess,
                                   data_loss, phys_loss_data)                   
     # phys trainer
     initial_num_epoch, initial_sampler = Setup_random_initials(config['train_initial'], obj_phys)
     phys_process, phys_loss = Setup_phys_trainer(obj_phys, time_refine_ratio=2.)
     phys_train_obj = Phys_trainer(initial_sampler, initial_num_epoch, phys_process.Input_preprocess, 
                                   phys_process.Output_postprocess, phys_loss) 
     # training
     epochs = config['train']['iterations']
     log_list, model = Training_model(epochs, model, optimizer, scheduler, data_train_obj, 
                                      phys_train_obj, device=device, use_tqdm=True, 
                                      write_log=log_run.Write_log, save_checkpoint=save_ckpt)      
     # save log list  
     Save_list_dict_excel(output_dir, 'training', log_list)
     # finish 
     return
     


# Tesing trained model (evolution)
def Main_testing(config: yaml.Loader):
     # device
     device = Setup_device()  
     # root, input, output dir
     input_dir = config['dir']['root_dir'] + config['dir']['input_dir'] 
     output_dir = config['dir']['root_dir'] + config['dir']['output_dir'] 
     os.makedirs(output_dir, exist_ok=True)

     # phys_obj, determining the equation type
     obj_phys = Setup_qme(config['phys'], device, input_dir)

     # reset time_size for eavl_data
     if 'eval_time_size' in config['eval_data']:
          obj_phys.dim_time  = config['eval_data']['eval_time_size']
     # Data-trainer for eval
     data_process = Transform_timewise(obj_phys.dim_time)
     # data loss 
     data_loss = {'data_xy' : Loss_data() }
     # data loader for eval
     data_train_obj = Data_trainer(Setup_dataloader(config['eval_data'], input_dir, obj_phys, False),
                                   data_process.Input_preprocess, data_process.Output_postprocess, data_loss)        
     # load model
     state_dict = Setup_state_dict(config['eval_data'], output_dir)     
     model = Setup_FNO(config['model'], device, state_dict)   
     # Train     
     print('Test with validation dataset ... \n')  
     log_list = Testing_model(model, data_train_obj, device=device)   
     # save log dict
     Save_list_dict_excel(output_dir, 'validation', log_list)
     return
     



# main
def Main_func(config_path: str):
     '''
     config_path: 
          name of 'yaml' file
     '''     
     # config yaml
     with open(config_path, 'r') as stream:
          config = yaml.load(stream, Loader=yaml.SafeLoader)    

     # main     
     print('')
     print('Current time: ', datetime.datetime.now(), '\n')
     print('')
     print('Quantum Fourier Neural Operator, training, start ... \n')  
     Main_training(config)
     # Main_testing(config)
     print('\n', 'Quantum Fourier Neural Operator, training, finish ... \n')


     return




if __name__ == '__main__':
     Main_func('./qme_train.yaml')

