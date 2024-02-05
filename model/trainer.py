import torch
import torch.utils.data
from timeit import default_timer
from typing import Callable
from tqdm import tqdm



# cuda synchronize
def _get_device_time(device: torch.device) -> float:
     if device.type == 'cuda': torch.cuda.synchronize(device)   
     return default_timer()

# Base trainer class
class Trainer_base:
     def __init__(self, num_train_data: int, pre_process: Callable[[torch.Tensor], torch.Tensor],
                  post_process: Callable[[torch.Tensor], torch.Tensor]) -> None:
          self.num_train_data = num_train_data
          self.pre_process = pre_process if callable(pre_process) else None
          self.post_process = post_process if callable(post_process) else None
     
     def Train(self, *args, **kwargs):
          pass

     def Eval(self, *args, **kwargs):
          pass


class Data_trainer(Trainer_base):
     """Data trainer (using dataset)

     Arguments
     ----------
     data_loader:
          data loader for training
     pre_process:
          transform the input tensor before passing to model
     post_process:
          transform the output of model
     loss_dict_data:
          loss functions evaluated from dataset (L_data)
     loss_dict_phys:
          loss functions evaluated from physics-informed (L_phys)
     """
     def __init__(self, data_loader: torch.utils.data.DataLoader,
                  pre_process: Callable[[torch.Tensor], torch.Tensor],
                  post_process: Callable[[torch.Tensor], torch.Tensor],
                  loss_dict_out_y: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                  loss_dict_out_x0: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}) -> None:
          super().__init__(len(data_loader), pre_process=pre_process, post_process=post_process)
          self.data_loader = data_loader
          self.loss_dict_data = loss_dict_out_y
          self.loss_dict_phys = loss_dict_out_x0 if isinstance(loss_dict_out_x0, dict) else {}

     # train, one epoch
     def Train(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) \
          -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, float]]:        
          # settings
          zero = torch.zeros(1, device=device)
          sum_losses = { loss_key : 0. for loss_key in self.loss_dict_data.keys()}
          sum_losses = sum_losses | { loss_key : 0. for loss_key in self.loss_dict_phys.keys()}        
          # train  
          for x_0, y in self.data_loader:
               x_0, y = x_0.to(device), y.to(device)
               optimizer.zero_grad() 
               # pre-process
               if self.pre_process is not None:
                    out = model(self.pre_process(x_0)) 
               else: 
                    out = model(x_0)
               # post-process
               if self.post_process is not None: 
                    out = self.post_process(out)
               # losses (x_out, y_data)
               val_losses = {}
               for loss_str, loss_func in self.loss_dict_data.items():
                    val_losses[loss_str] = loss_func(out, y)
                    sum_losses[loss_str] += val_losses[loss_str].item() / self.num_train_data
               # losses (x_out, x_0)
               for loss_str, loss_func in self.loss_dict_phys.items():
                    val_losses[loss_str] = loss_func(out, x_0)
                    sum_losses[loss_str] += val_losses[loss_str].item() / self.num_train_data
               # backward
               val_total = zero
               for val in val_losses.values():
                    val_total = val_total + val
               val_total.backward()               
               optimizer.step()
               del out
          # for log       
          log_dict = sum_losses   
          # return
          return model, optimizer, log_dict
     
     # eval
     def Eval(self, model: torch.nn.Module, device: torch.device) -> list[dict]:    
          '''
          Return: list of metrics for each datapoint. 
          Last term: average value.
          '''
          log_list = []
          # settings          
          sum_losses = { loss_key : 0. for loss_key in self.loss_dict_data.keys()}
          sum_losses = sum_losses | { loss_key : 0. for loss_key in self.loss_dict_phys.keys()}             
          # eval
          with torch.no_grad():  
               for x_0, y in self.data_loader:
                    x_0, y = x_0.to(device), y.to(device)
                    # pre-process
                    if self.pre_process is not None:
                         out = model(self.pre_process(x_0)) 
                    else: 
                         out = model(x_0)
                    # post-process
                    if self.post_process is not None: 
                         out = self.post_process(out)
                    # losses (x_out, y_data)
                    val_losses = {}
                    for loss_str, loss_func in self.loss_dict_data.items():
                         val_losses[loss_str] = loss_func(out, y).item()   
                         sum_losses[loss_str] += val_losses[loss_str] / self.num_train_data      
                    # losses (x_out, x_0)
                    for loss_str, loss_func in self.loss_dict_phys.items():
                         val_losses[loss_str] = loss_func(out, x_0).item()                 
                         sum_losses[loss_str] += val_losses[loss_str] / self.num_train_data      
                    del out
                    log_list.append(val_losses)
          # sum of losses
          log_list.append(sum_losses)
          # return
          return log_list

     


class Phys_trainer(Trainer_base):
     """Physics trainer (using random initials)
     
     Arguments
     ----------
     random_sampler:
          random sampler of initials
     num_per_epoch:
          number of random samples in each epoch
     pre_process:
          transform the input tensor before passing to model
     post_process:
          transform the output of model
     loss_dict_phys:
          loss functions evaluated from physics-informed (L_phys)
     """
     def __init__(self, random_sampler: Callable[[], torch.Tensor], num_per_epochs: int,
                  pre_process: Callable[[torch.Tensor], torch.Tensor],
                  post_process: Callable[[torch.Tensor], torch.Tensor],
                  loss_dict_out_x0: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]) -> None:
          super().__init__(num_per_epochs, pre_process=pre_process, post_process=post_process)
          self.random_sampler = random_sampler
          self.loss_dict_phys = loss_dict_out_x0
          
     # train, one epoch
     def Train(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) \
          -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, float]]:                  
          # settings
          zero = torch.zeros(1, device=device)
          sum_losses = { loss_key : 0. for loss_key in self.loss_dict_phys.keys()}
          # train
          for _ in range(self.num_train_data):
               x_init = self.random_sampler().to(device)
               optimizer.zero_grad() 
               # pre process 
               if self.pre_process is not None:
                    x_pred = model(self.pre_process(x_init))        
               else:
                    x_pred = model(x_init)
               # post process
               if self.post_process is not None: x_pred = self.post_process(x_pred)                                                       
               # phys loss, losses (x_out, x_0)
               val_losses = {}
               for loss_str, loss_func in self.loss_dict_phys.items():
                    val_losses[loss_str] = loss_func(x_pred, x_init)
                    sum_losses[loss_str] += val_losses[loss_str].item() / self.num_train_data
               # backward
               val_total = zero
               for val in val_losses.values():
                    val_total = val_total + val
               val_total.backward()               
               optimizer.step()
               del x_pred      
          # for log
          log_dict = sum_losses
          return model, optimizer, log_dict
     

     # eval
     def Eval(self, model: torch.nn.Module, device: torch.device) -> dict[str, list[float]]:                  
          '''
          Return: list of metrics for each datapoint. 
          Last term: average value.
          '''
          log_list = []
          # settings          
          sum_losses = { loss_key : 0. for loss_key in self.loss_dict_phys.keys()}             
          # eval
          with torch.no_grad():  
               for x_0, y in self.data_loader:
                    x_0, y = x_0.to(device), y.to(device)
                    # pre-process
                    if self.pre_process is not None:
                         out = model(self.pre_process(x_0)) 
                    else: 
                         out = model(x_0)
                    # post-process
                    if self.post_process is not None: 
                         out = self.post_process(out)               
                    # losses (x_out, x_0)
                    val_losses = {}
                    for loss_str, loss_func in self.loss_dict_phys.items():
                         val_losses[loss_str] = loss_func(out, x_0).item()                 
                         sum_losses[loss_str] += val_losses[loss_str] / self.num_train_data      
                    del out
                    log_list.append(val_losses)
          # sum of losses
          log_list.append(sum_losses)
          # return
          return log_list
     




# save checkpoints
def _save_ckpt(ep: int | None, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
               scheduler: torch.optim.lr_scheduler.MultiStepLR, 
               save_func: Callable[[dict], None] | None) -> None:
     if save_func is not None:
          state_dict = {'model' : model.state_dict(), 'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict() }
          save_func(ep, state_dict)
     return


# round float in log_dict
def _log_dict_float_round(log_dict: dict[str, float | int]) -> str:
     N = len(log_dict)
     res = '{'
     for i, (k, v) in enumerate(log_dict.items()):
          if isinstance(v, float):
               res += f'{k}:{v : .5f}'
          else:
               res += f'{k}:{v}'
          if i < N - 1: res += ', '
     res += '}'
     return res

# write log during training
def _write_log_tqdm(write_out: Callable[[str], None], log_dict: dict[str, float]) -> str:
     # pbar.set_description('[' f'data_loss:{log_dict[data_trainer.data_loss_str]: .5f}' ']')
     # log dict str
     N = len(log_dict)
     res = '{'
     for i, (k, v) in enumerate(log_dict.items()):
          res += f'{k}:{v : .5f}'
          if i < N - 1: res += ', '
     res += '}'
     write_out(res)
     # pbar.set_postfix({'data_loss': sum_data, 'phys_loss': sum_phys})                             
     return res



def Training_model(iterations: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler.LRScheduler,
                   data_part: Data_trainer | None = None, phys_part: Phys_trainer | None = None, 
                   device: torch.device = 0, use_tqdm: bool = True, write_log = None,
                   save_checkpoint = None) -> tuple[list[dict], torch.nn.Module]:   
     """ Train the model datasets (and random initials)

     Arguments
     ----------
     iterations: int
          epochs
     model, optimizer, scheduler: ...
     data_part: Data_trainer
          training dataset
     phys_part: Phys_trainer
          random initial set
     device: ...
     use_tqdm:
          enable tqdm during training
     write_log:
          log pannel
     save_checkpoint:
          checkpoints
     """   
     # train with data / physics
     enable_train_data = isinstance(data_part, Data_trainer)
     enable_train_phys = isinstance(phys_part, Phys_trainer)
     if not enable_train_data and not enable_train_phys: 
          print('\nWarning: cannot find any training datasets.\n')          
          return
     # training settings
     model.train()    
     # output
     log_list = []
     # iterations
     if use_tqdm: 
          pbar = tqdm(range(iterations + 1), dynamic_ncols=True, smoothing=0.05)
          write_out = pbar.set_description
     else:
          pbar = range(iterations + 1)
          write_out = print
     # with torch.autograd.set_detect_anomaly(True):
     for ep in pbar:
          # timer    
          t1 = _get_device_time(device)
          # train with function data
          if enable_train_data:
               model, optimizer, log_dict = data_part.Train(model, optimizer, device)  
          else:
               log_dict = {}        
          # train with physics data
          if enable_train_phys:
               model, optimizer, log_dict_phys = phys_part.Train(model, optimizer, device)
               log_dict.update(log_dict_phys)
          # scheduler      
          scheduler.step()
          # timer       
          t2 = _get_device_time(device)
          # tqdm / print
          _write_log_tqdm(write_out, log_dict)
          # log output     
          log_dict['total_loss'] = sum([val for val in log_dict.values()])
          log_dict['time cost'] = t2 - t1                          
          log_dict['epoch'] = ep + 1
          log_list.append(log_dict)
          # save log
          if write_log is not None: write_log(_log_dict_float_round(log_dict))
          # save model
          _save_ckpt(ep, model, optimizer, scheduler, save_checkpoint)
     # save model after all epochs
     _save_ckpt('last', model, optimizer, scheduler, save_checkpoint)
     # return 
     return log_list, model



def Testing_model(model: torch.nn.Module, data_part: Data_trainer, device: torch.device = 0) -> list[dict]:  
     """ Test the model with datasets.

     Arguments
     ----------
     model: ...
     data_part: Data_trainer
          training dataset
     device: ...
     """    
     # test with data / physics
     enable_test_data = isinstance(data_part, Data_trainer)
     if not enable_test_data: 
          print('\nWarning: cannot find any testing datasets.\n')          
          return
     # settings
     model.eval()     
     with torch.no_grad():
          # timer   
          if device == 0: torch.cuda.synchronize(device)       
          t1 = default_timer()
          # test with function data
          log_list = []
          if enable_test_data:
               log_list += data_part.Eval(model, device)               
          # timer
          if device == 0: torch.cuda.synchronize(device)       
          t2 = default_timer()          
          # log output     
          log_list.append({ 'time cost' : t2 - t1  })   
     return log_list             
     


# Relvative Lp norm, time-wise (last index)
def _Lp_rel_timewise(x_pred: torch.Tensor, y_true: torch.Tensor, 
                     p_ord: float = 2) -> torch.Tensor:
     batch = x_pred.shape[0]
     Nt = x_pred.shape[-1]
     x_pred = x_pred.reshape(batch, -1, Nt)
     y_true = y_true.reshape(batch, -1, Nt)
     res_up = torch.linalg.vector_norm(x_pred - y_true, ord=p_ord, dim=-2)
     res_down = torch.linalg.vector_norm(y_true, ord=p_ord, dim=-2)  
     return torch.mean(res_up / res_down)
     
# Relative Lp norm    
def _Lp_rel(x_pred: torch.Tensor, y_true: torch.Tensor, p_ord: float = 2) -> torch.Tensor:
     batch = x_pred.shape[0]
     res = torch.linalg.vector_norm(x_pred.reshape(batch,-1) - y_true.reshape(batch,-1), ord=p_ord, dim=-1)
     y_norm = torch.linalg.vector_norm(y_true.reshape(batch,-1), ord=p_ord, dim=-1)
     return torch.mean(res / y_norm)



def Get_default_loss(timewise: bool = True):
     """
     Get loss functions     
     """
     if timewise:
          return lambda x, y: _Lp_rel_timewise(x, y, p_ord=2)
     else: 
          return lambda x, y: _Lp_rel(x, y, p_ord=2) 

 


def PDE_loss(Dt_rho: torch.Tensor, L_rho: torch.Tensor, t_interval: float,
             loss_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
     """Compute the residual of PDE

     Arguments
     ----------
     Dt_rho: tensor
          left-hand side (dt) of PDE
     L_rho: tensor
          right-hand side of PDE
     t_interval: float, default is 1.
          upper limit of training time
     loss_kernel:
          kernel of loss function     
     """
     Dt_rho /= t_interval
     if Dt_rho.shape[-1] != L_rho.shape[-1]:
          offsets = int((L_rho.shape[-1] - Dt_rho.shape[-1]))
          pde_loss = loss_kernel(Dt_rho, L_rho[..., :-offsets]) 
     else:
          pde_loss = loss_kernel(Dt_rho, L_rho) 
     return pde_loss


# forward finite difference method
def PDE_loss_Dt_FDM(rho_xt: torch.Tensor, accu: int = 2) -> torch.Tensor:
     """
     Forward finite difference method to calculate d_t rho(t)     
     """
     Nt = rho_xt.shape[-1]
     dt = 1. / Nt
     if accu == 1:
          Dt_rho = rho_xt[..., 1:] - rho_xt[..., :-1]
     elif accu == 2:
          Dt_rho = -0.5 * rho_xt[..., 2:] + 2. * rho_xt[..., 1:-1] - 1.5 * rho_xt[..., :-2]
     elif accu == 3:
          Dt_rho = (1./3.) * rho_xt[..., 3:] - (3./2.) * rho_xt[..., 2:-1] + \
               3. * rho_xt[..., 1:-2] - (11./6.) * rho_xt[..., :-3]
     elif accu == 4:
          Dt_rho = (-1./4.) * rho_xt[..., 4:] + (4./3.) * rho_xt[..., 3:-1] - \
               3. * rho_xt[..., 2:-2] + 4. * rho_xt[...,1:-3] - (25./12.) * rho_xt[...,:-4]
     else:
          raise ValueError(f'FDM, accuracy {accu} is not supported')
     return Dt_rho / dt
    




