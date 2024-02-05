import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable



class FNO(nn.Module):
    """Generic Neural Operator for N-dim function.

    Arguments
    ----------
    lifting: MLP
        lifting part of model
    kernel_block: Fourier_block
        Fourier layers of model
    projection: MLP
        projection part of model
    """
    def __init__(self, lifting: nn.Module, kernel_block: nn.Module, projection: nn.Module) -> None:
        super().__init__()     
        self.lifting = lifting
        self.kernel_block = kernel_block
        self.projection = projection
        

    def forward(self, x):
        # lifting
        x = self.lifting(x)
        # fourier layers
        x = self.kernel_block(x)
        # projection
        x = self.projection(x)
        # return
        return x
    



def _get_act_real(act: str):
    '''
    Real-valued activation function.
    '''
    # func = getattr(F, act)
    if act == 'tanh':
        func = F.tanh
    elif act == 'relu':
        func = F.relu
    else:
        raise ValueError(f'_get_act_real, {act} is not supported')
    return func


# complex relu (not in-place)
def _complex_relu(input: torch.Tensor) -> torch.Tensor:
    '''
    Complex ReLU = relu(real) + 1j*relu(imag)
    '''
    return F.relu(input.real).type(input.dtype) + 1.j*F.relu(input.imag).type(input.dtype)


# func(a+i*b) = func(a) + i*func(b)
def _elementwise_complex_act(input: torch.Tensor, func_core) -> torch.Tensor:
    return func_core(input.real).type(input.dtype) + 1.j*func_core(input.imag).type(input.dtype)



# Activation function
def Get_activation(act: str):
    """Setup activation function for complex-valued function.

    Arguments
    ----------
    act: string
        Name of activation function    
    """
    if act == 'relu':
        func = _complex_relu
    else:
        func_real = _get_act_real(act)
        func = lambda x: _elementwise_complex_act(x, func_real)
    return func






# Multilayer perceptron
class MLP_base(nn.Module):    
    """Generic structure of MLP

    Arguments
    ----------
    main_layer_type: function
        Create main layers with given in/out channels
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_layers: int
        Number of total layers
    hidden_channels: int
        Number of hidden channels
    non_linearity:  
        Activation function
    pre_process: function, optional
        Transform the input tensor before passing to layers
    post_process: function, optional
        Transform the output tensor of layers before return    
    """   
    def __init__(self, main_layer_type: Callable[[int, int], nn.Module], 
                in_channels: int, out_channels: int, n_layers: int = 1, 
                hidden_channels: int | None = None, non_linearity = F.gelu,    
                pre_process: Callable[[torch.Tensor], torch.Tensor] | None = None, 
                post_process: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()
        in_features = in_channels
        out_features = in_channels if out_channels is None else out_channels
        if isinstance(hidden_channels, int) and hidden_channels > 0:
            num_layers = n_layers if n_layers > 2 else 2
            hidden_features = hidden_channels                 
        else:
            num_layers = 1
            hidden_features = 0    
        # processes
        self.pre_process = pre_process if callable(pre_process) else None
        self.post_process = post_process if callable(post_process) else None
        # main layers
        self.fcs = nn.ModuleList()
        # main layers          
        for i in range(num_layers):
            if i == 0 and i == (num_layers - 1):
                self.fcs.append(main_layer_type(in_features, out_features))
            elif i == 0:
                self.fcs.append(main_layer_type(in_features, hidden_features))
            elif i == (num_layers - 1):
                self.fcs.append(main_layer_type(hidden_features, out_features))                         
            else:
                self.fcs.append(main_layer_type(hidden_features, hidden_features))           
        # act
        self.non_linearity = non_linearity  
        

    def forward(self, x):
        num_layers = len(self.fcs)
        # pre process
        if callable(self.pre_process): x = self.pre_process(x)
        # main layers
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < num_layers - 1:
                x = self.non_linearity(x)
        # post processes
        if callable(self.post_process): x = self.post_process(x)
        return x





# Convolution MLP
class MLP_conv(MLP_base):     
    def __init__(self, in_channels: int, out_channels: int, n_layers: int = 1,
                hidden_channels: int | None = None, non_linearity = F.gelu,                    
                dim_conv: int = 0, pre_process: Callable[[torch.Tensor], torch.Tensor] | None = None, 
                post_process: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        # layer type                    
        if dim_conv < 1 or dim_conv > 3: 
            raise ValueError(f'MLP connection, does not support {dim_conv}d-convolution layer. ')      
        else:
            conv_func = getattr(nn, f"Conv{dim_conv}d")
            conv_type = lambda ax, bx : conv_func(in_channels=ax, out_channels=bx, 
                                                  kernel_size=1, dtype=torch.cfloat)
        # init
        super().__init__(conv_type, in_channels, out_channels, n_layers, hidden_channels, 
                        non_linearity, pre_process=pre_process, post_process=post_process)


     

def Permute_channel_fourier(x: torch.Tensor, fourier_dim: int) -> torch.Tensor:
    '''
    (batch, x, y, channel) -> (batch, channel, x, y)
    '''
    x_dim = len(x.shape)
    ft_dim = fourier_dim
    c_dim = 1 
    res = tuple(range(0, x_dim - ft_dim - c_dim)) 
    res += tuple(range(x_dim - c_dim, x_dim)) 
    res += tuple(range(x_dim - c_dim - ft_dim, x_dim - c_dim))
    return x.permute(*res) 


def Permute_fourier_channel(x: torch.Tensor, fourier_dim: int) -> torch.Tensor:
    '''
    (batch, channel, x, y) -> (batch, x, y, channel)
    '''
    x_dim = len(x.shape)
    ft_dim = fourier_dim
    c_dim = 1 
    res = tuple(range(0, x_dim - ft_dim - c_dim)) 
    res += tuple(range(x_dim - ft_dim, x_dim)) 
    res += tuple(range(x_dim - ft_dim - c_dim, x_dim - ft_dim))
    return x.permute(*res) 
