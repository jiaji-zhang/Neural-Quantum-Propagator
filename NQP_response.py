import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import yaml, pandas
import torch
import torch.nn.functional as F

from train import Save_list_dict_excel
from model import QME, Physop


from NQP_setups import Setup_device, Setup_qme, Setup_FNO, Setup_state_dict, Transform_timewise

# physical operators in tensor format, both A_times and A_right
def Setup_resp_physop(file_op: str, input_dir: str, obj_phys: QME) -> Physop:
     if not input_dir.endswith('/'): input_dir += '/'
     if not file_op.endswith('.pt'): file_op += '.pt'
     Op_times = Physop(torch.load(input_dir + file_op).to(obj_phys.device)  )
     return Op_times
        

# model evolution
def Model_evolution(rho_init: torch.Tensor, evolve_func, truncate_ratio : float = 1.) \
     -> tuple[torch.Tensor, torch.Tensor]:
     '''
     evolve_func: rho_init (sample, x_sys) -> rho_time (sample, x_sys, time)     
     ratio: keep first several elements in time-dim 
     return rho_time[...,-1] as next rho_init, rho_time after truncation
     '''
     rho_time = evolve_func(rho_init)
     if truncate_ratio > 0. and truncate_ratio < 1.:
          tdim = int(truncate_ratio * rho_time.shape[-1])
          rho_time = rho_time[..., :tdim]
     rho_init = rho_time[..., -1]
     return rho_init, rho_time


     

# Evaluate population dynamics using trained model
# Read initial values from validation dataset
def Main_population(config: yaml.Loader, state_label: int) -> None:
     # device
     device = Setup_device()
     dtype_phys = torch.cfloat     
     # root, input, output dir
     input_dir = config['dir']['root_dir'] + config['dir']['input_dir'] 
     output_dir = config['dir']['root_dir'] + config['dir']['output_dir'] 
     os.makedirs(output_dir, exist_ok=True)

     # phys_obj, determining the equation type
     obj_phys = Setup_qme(config['phys'], device, input_dir)

     # Data-trainer for eval
     data_process = Transform_timewise(obj_phys.dim_time)   
     
     # rho_evolve (1, x_sys, time)
     evolve_name = config['pure_state_evolve']['file_name']
     if not evolve_name.endswith('.pt'): evolve_name += '.pt'
     rho_evolve = torch.load(input_dir + evolve_name).to(obj_phys.device) 
     rho_evolve = rho_evolve[state_label, ...].unsqueeze(0) 
     rho_init = rho_evolve[..., 0] 
     state_label = 'No_' + str(state_label)       

     # extract data
     extract_pop = lambda x: obj_phys.Extract_population(x)

     # create model, optimizers
     state_dict = Setup_state_dict(config['model'], output_dir, load_strict=True)
     model = Setup_FNO(config['model'], dtype_phys, device, state_dict)        
     # evolve of model
     def evolve_func(x):
          x = data_process.Input_preprocess(x)
          x_out = model(x)
          x_out = data_process.Output_postprocess(x_out)
          return x_out
     
     # evaluate response function   
     print('Evaluate population dynamics with trained model ...')  
     N_t_iter = 6 
     model.eval()     
     with torch.no_grad():
          for n in range(N_t_iter):
               rho_init, rho_temp = Model_evolution(rho_init, evolve_func)
               rho_temp = extract_pop(rho_temp)        
               if n == 0:
                    rho_pred = rho_temp
               else:
                    rho_pred = torch.cat((rho_pred, rho_temp[..., 1:]), dim=-1)                

     # rho_pred (states, time)     
     rho_pred = rho_pred.squeeze(0).real
     rho_evolve = extract_pop(rho_evolve).squeeze(0).real     
    
     # compute losses
     population_loss = []
     t_length = min(rho_pred.shape[-1], rho_evolve.shape[-1])
     for t in range(t_length):
          loss_t = {}
          N_s_calc = 7 # only focus on state 0,1,2
          for n in range(N_s_calc):
               loss_t['p_' + str(n) + '_fno'] = rho_pred[n][t].item()
               loss_t['p_' + str(n) + '_rk4'] = rho_evolve[n][t].item()
          population_loss.append(loss_t)

     
     
     # save to file    
     Save_list_dict_excel(output_dir, 'population_' + state_label, population_loss) 
          
     return
     

     
# Evaluate linear response function using trained model
def Main_response_1st(config: yaml.Loader, state_label: int) -> None:
     # device
     device = Setup_device()
     dtype_phys = torch.cfloat     
     # root, input, output dir
     input_dir = config['dir']['root_dir'] + config['dir']['input_dir'] 
     output_dir = config['dir']['root_dir'] + config['dir']['output_dir'] 
     os.makedirs(output_dir, exist_ok=True)

     # phys_obj, determining the equation type
     obj_phys = Setup_qme(config['phys'], device, input_dir)
     # Data-trainer for eval
     data_process = Transform_timewise(obj_phys.dim_time)     

     # initial condition data for response
     resp_init_name = config['resp_data']['file_resp_init']
     if not resp_init_name.endswith('.pt'): resp_init_name += '.pt'
     resp_init = torch.load(input_dir + resp_init_name).to(obj_phys.device) 
     resp_init = resp_init[state_label, :].unsqueeze(0)
     # linear response
     resp_init_name = config['resp_data']['file_resp_1st']
     if not resp_init_name.endswith('.pt'): resp_init_name += '.pt'     
     resp_1st_data = torch.load(input_dir + resp_init_name).to(obj_phys.device)
     # truncate
     resp_trun = 100
     resp_1st_data = resp_1st_data[state_label, :resp_trun].unsqueeze(0)    
     state_label = str(state_label)

     # phys operators
     Op_times = Setup_resp_physop(config['resp_data']['physop_times'], input_dir, obj_phys)
     Op_right = Setup_resp_physop(config['resp_data']['physop_last_trace'], input_dir, obj_phys)     

     # create model, optimizers
     state_dict = Setup_state_dict(config['model'], output_dir, load_strict=True)
     model = Setup_FNO(config['model'], dtype_phys, device, state_dict)        
     # evolve of model
     def evolve_func(x):
          x_out = model(data_process.Input_preprocess(x))
          return data_process.Output_postprocess(x_out)

     # evaluate response function   
     print('Evaluate response functions with trained model ...')  
     model.eval() 
     N_iter = 3    
     with torch.no_grad():
          # apply first operator
          resp_init = Op_times.Apply_to(resp_init)
          # evolve (1, Ns) -> (1, Ns, t1) or (1, Nh, Ns) -> (1, Nh, Ns, t1)
          for n in range(N_iter):
               resp_init, resp_temp = Model_evolution(resp_init, evolve_func)
               if n == 0:
                    resp_1st_pred = resp_temp
               else:
                    resp_1st_pred = torch.cat((resp_1st_pred, resp_temp[..., 1:]), dim=-1)
          # permute (1, Ns, t1) -> (1, t1, Ns) or (1, Nh, Ns, t1) -> (1, t1, Nh, Ns)
          Nd = len(resp_1st_pred.shape)
          resp_1st_pred = resp_1st_pred.permute(0, Nd - 1, *range(1, Nd - 1))
          # trace (1, t1, Ns) -> (1, t1)
          resp_1st_pred = Op_right.Trace_last_dim(resp_1st_pred).imag          

     # (1, t_1) -> (t_1)
     resp_1st_pred = resp_1st_pred.squeeze(0) 
     resp_1st_data = resp_1st_data.squeeze(0)

     # compute losses, time domain     
     t_length = min(resp_1st_pred.shape[-1], resp_1st_data.shape[-1])
     resp_1st_pred = resp_1st_pred[:t_length]
     resp_1st_data = resp_1st_data[:t_length] 
     resp_1st_time = torch.as_tensor([0.02 * n for n in range(t_length)])
     # time domain    
     time_losses = []
     for t in range(t_length):
          loss_t = {'t1' : resp_1st_time[t].item()}
          loss_t['Time_fno'] = resp_1st_pred[t].item()
          loss_t['Time_rk4'] = resp_1st_data[t].item()
          time_losses.append(loss_t)

     # window
     wind_func = True
     if wind_func:
          wind_data = torch.hamming_window(t_length, device=device)
          resp_1st_pred = resp_1st_pred * wind_data
          resp_1st_data = resp_1st_data * wind_data
          

     # zero pad
     zero_pad_level = 100
     if zero_pad_level > 0:
          freq_length = t_length + zero_pad_level
          resp_1st_pred = F.pad(resp_1st_pred, (0, zero_pad_level), 'constant', 0)
          resp_1st_data = F.pad(resp_1st_data, (0, zero_pad_level), 'constant', 0)
     else:
          freq_length = t_length
     # fourier domain     
     resp_1st_freq, _ = torch.sort(torch.fft.fftfreq(freq_length) * torch.pi)
     resp_1st_pred_fft = torch.fft.fftshift(torch.fft.fft(resp_1st_pred, dim=-1)).imag
     resp_1st_data_fft = torch.fft.fftshift(torch.fft.fft(resp_1st_data, dim=-1)).imag
     resp_1st_pred_fft /= torch.abs(torch.max(resp_1st_pred_fft))
     resp_1st_data_fft /= torch.abs(torch.max(resp_1st_data_fft))

     freq_nums = 100
     freq_starts = freq_length - freq_nums
     slcx = slice(freq_starts//2, -freq_starts//2)
     resp_1st_freq = resp_1st_freq[slcx]
     resp_1st_pred_fft = resp_1st_pred_fft[slcx]
     resp_1st_data_fft = resp_1st_data_fft[slcx]        
     
     
     freq_losses = []
     for t in range(resp_1st_freq.shape[0]):
          loss_t = {'omega_1' : resp_1st_freq[t].item()}
          loss_t['Freq_fno'] = resp_1st_pred_fft[t].item()
          loss_t['Freq_rk4'] = resp_1st_data_fft[t].item()
          freq_losses.append(loss_t)

     # save to file    
     file_name = output_dir + 'resp_1st_' + state_label + '.xlsx'
     with pandas.ExcelWriter(file_name) as writer:            
          pandas.DataFrame(freq_losses).to_excel(writer, sheet_name='Freq')
          pandas.DataFrame(time_losses).to_excel(writer, sheet_name='Time')
     return
     

# Evaluate second order response function using trained model
def Main_response_2nd(config: yaml.Loader, state_label: int) -> None:
     # device
     device = Setup_device()
     dtype_phys = torch.cfloat     
     # root, input, output dir
     input_dir = config['dir']['root_dir'] + config['dir']['input_dir'] 
     output_dir = config['dir']['root_dir'] + config['dir']['output_dir'] 
     os.makedirs(output_dir, exist_ok=True)

     # phys_obj, determining the equation type
     obj_phys = Setup_qme(config['phys'], device, input_dir)

     # Data-trainer for eval
     data_process = Transform_timewise(obj_phys.dim_time)    

     # initial condition data for response
     resp_init_name = config['resp_data']['file_resp_1st']
     if not resp_init_name.endswith('.pt'): resp_init_name += '.pt'   
     resp_init = torch.load(input_dir + resp_init_name).to(obj_phys.device) 
     resp_init = resp_init[state_label, :].unsqueeze(0)
     resp_init_name = config['resp_data']['file_resp_2nd']
     if not resp_init_name.endswith('.pt'): resp_init_name += '.pt'   
     resp_2nd_data = torch.load(input_dir + resp_init_name).to(obj_phys.device)
     resp_2nd_data = resp_2nd_data[state_label, ...]
     state_label = str(state_label)

     # phys operators
     Op_times = Setup_resp_physop(config['resp_data']['physop_times'], input_dir, obj_phys)
     Op_right = Setup_resp_physop(config['resp_data']['physop_last_trace'], input_dir, obj_phys)     

     # create model, optimizers
     state_dict = Setup_state_dict(config['model'], output_dir, load_strict=True)
     model = Setup_FNO(config['model'], dtype_phys, device, state_dict)        
     # evolve of model
     def evolve_func(x):
          x_out = model(data_process.Input_preprocess(x))
          return data_process.Output_postprocess(x_out)
     
     

     # evaluate response function   
     print('Evaluate response functions with trained model ...')  
     N1_iter = 6
     N2_iter = 6
     model.eval()
     with torch.no_grad():
          # apply first operator 
          resp_init = Op_times.Apply_to(resp_init)
          # evolve t_1, (1, Nh, Ns) -> (1, Nh, Ns, t1)     
          for n in range(N1_iter):
               resp_init, resp_temp = Model_evolution(resp_init, evolve_func)
               if n == 0:
                    resp_2nd_pred = resp_temp
               else:
                    resp_2nd_pred = torch.cat((resp_2nd_pred, resp_temp[..., 1:]), dim=-1)          
          # apply second operator (1, Nh, Ns, t1) -> (Nh, Ns, t1) -> (t1, Nh, Ns)  
          resp_2nd_pred = resp_2nd_pred.squeeze(0)
          Nd = len(resp_2nd_pred.shape)    
          resp_2nd_pred = resp_2nd_pred.permute(Nd - 1, *range(0, Nd - 1))           
          resp_2nd_pred = Op_times.Apply_to(resp_2nd_pred)
          # evolve t_2, (t1, Nh, Ns) -> (t1, Nh, Ns, t2) 
          resp_2nd_pred = [resp_2nd_pred[b].unsqueeze(0) for b in range(resp_2nd_pred.shape[0])]
          for b in range(len(resp_2nd_pred)):
               # batch 1: (1, Nh, Ns) 
               resp_init = resp_2nd_pred[b]     
               for n in range(N2_iter):
                    resp_init, resp_temp = Model_evolution(resp_init, evolve_func)
                    if n == 0:
                         resp_2nd_pred[b] = resp_temp
                    else:
                         resp_2nd_pred[b] = torch.cat((resp_2nd_pred[b], resp_temp[..., 1:]), dim=-1)
          resp_2nd_pred = torch.cat(resp_2nd_pred, dim=0)          
          # permute (t1, Nh, Ns, t2) -> (t1, t2, Nh, Ns)
          Nd = len(resp_2nd_pred.shape)                    
          resp_2nd_pred = resp_2nd_pred.permute(0, Nd - 1, *range(1, Nd - 1))  
          # trace (t1, t2)
          resp_2nd_pred = Op_right.Trace_last_dim(resp_2nd_pred).real
       
     # response function data in format (t1, t2, value)
     torch.save(resp_2nd_pred, output_dir + 'resp_2nd_time_fno_[' + state_label + '].pt')
     torch.save(resp_2nd_data, output_dir + 'resp_2nd_time_rk4_[' + state_label + '].pt')
             
     return
     



def Main_func(config_path: str):
     '''
     config_path: 
          name of 'yaml' file
     '''     
     # config yaml
     with open(config_path, 'r') as stream:
          config = yaml.load(stream, Loader=yaml.SafeLoader)  
     # main     
     state_label = 5
     Main_population(config, state_label)
     Main_response_1st(config, state_label)
     Main_response_2nd(config, state_label)     
     return

if __name__ == '__main__':
     Main_func('./qme_resp.yaml')
