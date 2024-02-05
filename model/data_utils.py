import os
import torch
import torch.utils.data

import numpy as np
import pandas

# Create data-loader from tensors 
def Create_dataloader(*x: tuple[torch.Tensor], batch_size: int = 1, 
                      shuffle: bool = True) -> torch.utils.data.DataLoader:
     """Generate data loader from tensors

     Arguments
     ----------
     x: tuple[tensors]
          data tensors
     batch_size: int
          batch size
     shufflt: bool
          'True' for training, False for validation.   
     """
     dataset = torch.utils.data.TensorDataset(*x)
     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
     return dataloader




def Insert_dimension(u0: torch.Tensor, Nb: int, loc: int = -1) -> torch.Tensor:     
     """
     Insert new dimension (b) to (x, y, ...)
     """     
     u0 = u0.unsqueeze(loc)
     shape_rep = [1]*len(u0.shape)
     shape_rep[loc] = Nb
     res = u0.repeat(shape_rep)
     return res



def _uniform_linspace(N, min=0., max=1., dtype=torch.float, device='cpu') -> torch.Tensor:
     return torch.tensor(np.linspace(min, max, N, endpoint=False), 
                         dtype=dtype, device=device)

def _repeat_dimension(tensor: torch.Tensor, dim: tuple[int] | list[int]) -> torch.Tensor:
     if not isinstance(dim, tuple): dim = tuple(dim)
     dim_res = (1,) * len(dim) + tuple(tensor.shape)
     dim_rep = dim + (1,) * len(tensor.shape)
     return tensor.reshape(dim_res).repeat(dim_rep)


def One_dim_grid(dim: int, dtype: torch.dtype = torch.float, device: torch.device = 0, 
                 dim_repeat: tuple[int] | None = None) -> torch.Tensor:
     """
     1D grid with repeated values: (Nt) ->(Nx, ..., Nt, 1)
     """
     grid = _uniform_linspace(dim, 0., 1., dtype, device)
     if isinstance(dim_repeat, tuple): grid = _repeat_dimension(grid, dim_repeat)
     grid = grid.unsqueeze(-1)
     return grid
   
   
# Save and load model's state-dict
def Save_module_state_dict(path_directory: str, file_name: str, state_dict: dict[str, object],
                           num_epoch: int | None = None, save_steps: int | None = None) -> None: 
     model_dir = path_directory
     if not model_dir.endswith('/'): model_dir += '/'    
     if isinstance(num_epoch, int) and isinstance(save_steps, int):
          if (save_steps > 0) and (num_epoch % save_steps == 0):
               model_dir += 'checkpionts/'
               model_name = model_dir + file_name + '_[' + str(num_epoch) + '].pt'
               os.makedirs(model_dir, exist_ok=True)     
               torch.save(state_dict, model_name)    
     elif isinstance(num_epoch, str) and num_epoch:
          model_name = model_dir + file_name + '_[' + num_epoch + '].pt'
          os.makedirs(model_dir, exist_ok=True)    
          torch.save(state_dict, model_name)    
          print('')
          print('Model is saved at %s \n' % model_name)   
     else:
          pass       
     return


def Load_module_state_dict(file_name: str) -> dict[str, object]:
     if not file_name.endswith('.pt'): file_name += '.pt'
     if not os.path.isfile(file_name):
          raise ValueError(f'{file_name}, file not found')  
     ckpt = torch.load(file_name)
     print('Model is loaded from %s \n' % file_name) 
     return ckpt
   



def Save_list_dict_excel(path_directory: str, file_name: str, list_dict: list[dict]) -> None:
     df = pandas.DataFrame(list_dict)
     if isinstance(path_directory, str) and path_directory: 
          if not path_directory.endswith('/'): path_directory += '/'    
          file_name = path_directory + file_name
     if not file_name.endswith('.xlsx'): file_name += '.xlsx'
     df.to_excel(file_name)
     print('List_of_dict is saved at %s' % file_name)
     return



