import random
import json
import torch

# model
from model import Get_activation, Permute_fourier_channel, Permute_channel_fourier
from model import MLP_conv, FNO, Fourier_block
from model import QME, Online_sample_qme

# data 
from model import One_dim_grid, Insert_dimension, Create_dataloader
from model import Save_module_state_dict, Load_module_state_dict, Get_default_loss, PDE_loss_Dt_FDM


def Setup_device(seed: int = 0) -> torch.device:
     """
     Initialize device
     """   
     if torch.cuda.is_available:
          device = torch.device('cuda:0')
          print(f'GPU available, use { device } ... \n')   
     else:
          device = torch.device('cpu')
          print('GPU unavailable, use cpu instead ... \n')
     # seed: random seed
     torch.manual_seed(seed)
     random.seed(seed)
     if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
     return device



class Text_log:
     """Log class during training
     
     Arguments
     ----------
     output_dir: str
          output directory
     file_name: str
          file name of log
     """
     def __init__(self, output_dir: str, file_name: str) -> None:
          self.output_dir = output_dir
          if not output_dir.endswith('/'): output_dir += '/'
          if not file_name.endswith('.txt'): file_name += '.txt'
          self.log_file = output_dir + file_name
          with open(self.log_file, 'w') as f_stream:
               pass

     def Write_log(self, x: str) -> None:
          if not isinstance(x, str): x = json.dumps(x)
          with open(self.log_file, "a") as f_stream:
               f_stream.write(x)
               f_stream.write('\n')
          return
     

# logs, checkpoints
def Setup_logs(output_dir: str):
     """
     Initialize log-class and checkpoints
     """
     # log
     log_file_name = 'logs_training'
     log_obj = Text_log(output_dir, log_file_name)
     # save checkpoints
     state_dict_save_name = 'state_dict'
     save_ckpt_step = '50'     
     save_ckpt = lambda num, state_dict: Save_module_state_dict(
          output_dir, state_dict_save_name, state_dict, num, save_ckpt_step) 
     return log_obj, save_ckpt


# load state_dict
def Setup_state_dict(config_train: dict, output_dir: str | None = None):
     """
     Load module's state-dict from file
     """
     file_state = output_dir + config_train['model_load']
     if not file_state.endswith('.pt'): file_state += '.pt'
     state_dict = Load_module_state_dict(file_state)             
     return state_dict



class Loss_data:
     """
     Loss function for data (L_data)
     """
     def __init__(self) -> None:
          self.loss_kernel = Get_default_loss(True)

     def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
          '''
          x: model's prediction \ 
          y: values in dataset
          '''    
          return self.loss_kernel(x, y) 

     
class Loss_initial:
     """
     Loss function for initial condition.
     """
     def __init__(self) -> None:    
          self.loss_kernel = Get_default_loss(False)
          

     def __call__(self, rho_time: torch.Tensor, rho_init: torch.Tensor) -> torch.Tensor:
          rho_pred = rho_time[..., 0]
          return self.loss_kernel(rho_pred, rho_init) 
    


class Loss_equation:
     '''
     Loss function for QME
     '''
     def __init__(self, t_interval: float, L_matrix: torch.Tensor, rhs_pde) -> None:
          self.loss_kernel = Get_default_loss(True)  
          self.t_interval = t_interval
          self.L_matrix = L_matrix 
          self.rhs_pde = rhs_pde      
     
     def _pde_lhs_rhs(self, rho_out: torch.Tensor) -> torch.Tensor:
          lhs_dt = PDE_loss_Dt_FDM(rho_out, 3) / self.t_interval
          rhs_L_rho = self.rhs_pde(rho_out)
          # keep tensor size matched
          if lhs_dt.shape[-1] != rhs_L_rho.shape[-1]:
               offsets = int((rhs_L_rho.shape[-1] - lhs_dt.shape[-1]))
               rhs_L_rho = rhs_L_rho[..., :-offsets]
          # compute
          return self.loss_kernel(lhs_dt, rhs_L_rho) 

     def __call__(self, rho_out: torch.Tensor, rho_init: torch.Tensor | None = None) -> torch.Tensor:
          return self._pde_lhs_rhs(rho_out)





def Setup_FNO(config_model: dict, device: torch.device, state_dict = None) -> torch.nn.Module:
     """
     Initialize FNO model
     """
     # parameters, lift/proj
     in_dim = 2 
     out_dim = 1 
     lift_proj_layers = config_model['lift_proj_layers'] 
     lift_proj_dim = config_model['lift_proj_dim']    
     # activation function
     act_func = Get_activation(config_model['act_function'])
     # parameters, kernel
     fourier_layers= config_model['fourier_layers']
     fourier_width = config_model['fourier_width']
     fourier_modes = config_model['fourier_modes']
     fft_order = len(fourier_modes)
     fft_pad_level = config_model['pad_level'] if 'pad_level' in config_model else None  
     # permute process
     to_channel_fourier = lambda x: Permute_channel_fourier(x, fft_order)
     to_fourier_channel = lambda x: Permute_fourier_channel(x, fft_order)
     # lifting projection
     lifting = MLP_conv(in_dim, fourier_width, lift_proj_layers, lift_proj_dim,  
                        act_func, dim_conv=fft_order, pre_process=to_channel_fourier)
     projection = MLP_conv(fourier_width, out_dim, lift_proj_layers, lift_proj_dim,
                           act_func, dim_conv=fft_order, post_process=to_fourier_channel)
     # kernel
     kernel_block = Fourier_block(fourier_layers, fourier_width, fourier_modes, 
                                  non_linearity=act_func,fourier_pad_level=fft_pad_level)
     # wrap as model
     model = FNO(lifting, kernel_block, projection).to(device)     
     if state_dict is not None:
          model.load_state_dict(state_dict['model'], strict=True)
     return model




def Setup_dataloader(config_data: dict, input_dir: str, obj_phys: QME,
                      mode_train: bool = True): 
     """
     Initialize data loader
     """
     file_name = input_dir + config_data['file_name']
     offset = config_data['offset'] if 'offset' in config_data else 0
     sample_size = config_data['num_sample']
     # time evolution (time independent Hamiltonian)
     data_read = obj_phys.Load_data_file(file_name, sample_size, offset)
     print('Load evolution data from ', file_name, '\n')  
     rho_init = data_read[..., 0]      
     if mode_train:
          batch_size = config_data['batch_size'] 
          shuffle = config_data['shuffle'] if 'shuffle' in config_data else False             
          evolve_loader = Create_dataloader(rho_init, data_read, batch_size=batch_size, shuffle=shuffle)
     else:
          evolve_loader = Create_dataloader(rho_init, data_read, batch_size=1, shuffle=False)
     return evolve_loader



def Setup_random_initials(config_random: dict | None, obj_phys: QME):
     """
     Initialize random initials.
     """
     initial_sample = config_random['initial_sample']
     batch_size = config_random['batch_size']
     initial_num_epoch = int(initial_sample / batch_size)
     initial_sampler = lambda : Online_sample_qme(batch_size, obj_phys.dim_sys, obj_phys.device) 
     return initial_num_epoch, initial_sampler




def Setup_qme(config_phys: dict, device: torch.device, input_dir: str) -> QME: 
     """
     Initialize class for QME
     """
     N_sys = config_phys['system_size'] 
     N_time = config_phys['time_nums']     
     L_csr_file = input_dir + config_phys['file_L_csr']
     obj_phys = QME(N_time, N_sys, L_csr_file, device=device)
     return obj_phys




class Transform_timewise:
     """
     Pre-process before passing to model.
     Post-process for model's output.
     """
     def __init__(self, time_size: int, refine_ratio: float = 1.) -> None:                                 
          # time size
          self.time_size = int(time_size * refine_ratio) if refine_ratio > 1. else time_size
          
     # time grid (t) -> (b,h,x,t,1)
     def _time_grid(self, other_size: tuple[int], dtype: torch.dtype, 
                    device: torch.device = 0) -> torch.Tensor:
          return One_dim_grid(self.time_size, dtype, device, other_size)

     # pre-process
     def Input_preprocess(self, rho_init: torch.Tensor) -> torch.Tensor:
          '''
          1. append time grid: (b,h,x) -> (b,h,x,t,2)
          2. permute: (b,h,x,t,2) -> (bt,h,x,2)
          '''
          # grid
          t_grid = self._time_grid(rho_init.shape, rho_init.dtype, rho_init.device)
          rho_init = Insert_dimension(rho_init, self.time_size, -1).unsqueeze(-1)
          rho_grid = torch.cat((rho_init, t_grid), dim=-1)
          # permute
          dim = len(rho_grid.shape)
          rho_grid = rho_grid.permute(0, dim-2, *tuple(range(1, dim-2)), dim-1)
          rho_grid = rho_grid.reshape(-1, *rho_grid.shape[2:])
          return rho_grid


     # post-process
     def Output_postprocess(self, rho_out: torch.Tensor) -> torch.Tensor:
          '''
          1. reshape (..., 1) -> (...)
          2. permute: (bt,...) -> (b,...,t)               
          '''
          rho_out = rho_out.squeeze(-1)          
          # permute (bt, ...) -> (b,...,t)
          batch_size = int(rho_out.shape[0] / self.time_size)
          rho_out = rho_out.reshape(batch_size, self.time_size, *rho_out.shape[1:])
          dim = len(rho_out.shape)
          rho_out = rho_out.permute(0, *tuple(range(2, dim)), 1)
          return rho_out



def Setup_data_trainer(obj_phys: QME):
     """
     Initialize data trainer
     """
     N_time = obj_phys.dim_time 
     # processes
     process_data = Transform_timewise(N_time)
     # losses
     data_loss = { 'data_xy' : Loss_data() }
     # phys loss, data part 
     phys_loss_data = {'data_pde' : Loss_equation(obj_phys.t_interval, obj_phys.L_matrix, obj_phys.PDE_residual) } 
     return process_data, data_loss, phys_loss_data



def Setup_phys_trainer(obj_phys: QME, time_refine_ratio: float = 1.):
     """
     Initialize phys trainer
     """
     # phys loss, online random samples   
     N_time = obj_phys.dim_time
     # processes
     process_phys = Transform_timewise(N_time, time_refine_ratio)
     # losses
     phys_loss = { 'phys_init' : Loss_initial(),
                   'phys_pde' :  Loss_equation(obj_phys.t_interval, obj_phys.L_matrix, obj_phys.PDE_residual)} 
     return process_phys, phys_loss


