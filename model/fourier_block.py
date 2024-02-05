import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable


# tensorly factorization
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


# Spectral convlution
class Spec_conv(nn.Module):
    """Spectral convolution, upper route of Fourier_layer

    Arguments
    ----------
    channels : int
        Number of channels for each Fourier mode
    n_modes: int or list[int]
        Number of Fourier modes in each direction 
    """
    def __init__(self, in_channels: int,n_modes: list[int]) -> None:
        super().__init__()
        self.in_channels = in_channels      
        # n_modes is the total number of modes kept along each dimension
        if isinstance(n_modes, int): n_modes = [n_modes]
        self.n_modes = n_modes
        self.fft_order = len(self.n_modes)    
        # scale (float) Kaiming Initialization: rescale by standard deviation   
        scale_factor = (1 / in_channels)**0.5 
        # Fourier weight tensors
        weight_shape = (in_channels, *n_modes)          
        self.weight = nn.Parameter(scale_factor * torch.randn(weight_shape, dtype=torch.cfloat))
        # add bias
        bias_shape = (in_channels,) + (1,) * self.fft_order
        self.bias = nn.Parameter(scale_factor * torch.randn(bias_shape, dtype=torch.cfloat))


    def forward(self, x):   
        batch_dims = list(x.shape[:(-self.fft_order - 1)])
        fft_dims = list(range(-self.fft_order, 0))
        fft_size = list(x.shape[-self.fft_order:])    
        fft_norm = 'ortho'    
        # fft and rearranges all frequencies into ascending order
        x = torch.fft.fftn(x, dim=fft_dims, norm=fft_norm)
        x = torch.fft.fftshift(x, dim=fft_dims) # dim=fft_dims
        # channel mixing: weight(in, out, x, y...); otherwise (in, x, y,...)
        x_slices = self._x_slices_truncate(x.shape, self.weight.shape, batch_dims)
        # apply weight, contract [x] with [weight]
        out_fft = torch.zeros([*batch_dims, self.in_channels, *fft_size], 
                              device=x.device, dtype=x.dtype)
        out_fft[x_slices] = self._contract(x[x_slices], self.weight)   
        # call fftshift or ifftshift before ifft
        out_fft = torch.fft.fftshift(out_fft, dim=fft_dims)
        x = torch.fft.ifftn(out_fft, dim=fft_dims, norm=fft_norm)
        # add bias
        return x + self.bias
    
   
    def _x_slices_truncate(self, x_shape: list[int], wt_shape: list[int], 
                           batch_dims: int) -> list[slice]:
        starts = [(size - min(size, n_mode)) for (size, n_mode) in 
                  zip(list(x_shape[-self.fft_order:]), list(wt_shape[-self.fft_order:]))]
        slices_x =  [slice(None)] * len(batch_dims) + [slice(None)] 
        slices_x += [slice(start//2, -start//2) if start else slice(start, None) for start in starts]
        return slices_x
    
    def _contract(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        order = len(x.shape)
        x_syms = list(einsum_symbols[:order])
        weight_syms = list(x_syms[(-self.fft_order - 1):]) 
        out_syms = x_syms.copy()        
        eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'
        if not torch.is_tensor(weight): weight = weight.to_tensor()    
        return torch.einsum(eq, x, weight)
    



# Fourier layer
class Fourier_block(nn.Module):
    """Spectral convolution, upper route of Fourier_layer

    Arguments
    ----------
    fourier_layers: int 
        number of each Fourier layer
    fourier_channels: int 
        channels of each Fourier layer
    fourier_modes: list[int] 
        modes of Fourier layer in each dimension   
    non_linearity: function
        Activation function
    fourier_pad_level: list[int], optional
        Zero_pad level in each dimension
    """
    def __init__(self, fourier_layers: int, fourier_channels: int, fourier_modes: tuple[int], 
                 non_linearity: Callable[[torch.Tensor], torch.Tensor] = F.gelu,           
                 fourier_pad_level: tuple[int] | None = None) -> None:
        # setup block
        super().__init__()
        # setup spec_conv        
        spconv_layers = fourier_layers
        spconv_width = fourier_channels
        spconv_modes = fourier_modes    
        fft_dim = len(spconv_modes)     
        dtype_input = torch.cfloat                  
        # Fourier layers
        spconv_setup = lambda : Spec_conv(spconv_width, spconv_modes)   
        self.spconvs = nn.ModuleList([spconv_setup() for _ in range(spconv_layers)])        
        # skip connections   
        conv_type = getattr(nn, f"Conv{fft_dim}d")
        skip_setup = lambda : conv_type(in_channels=spconv_width, out_channels=spconv_width, 
                                        kernel_size=1, dtype=dtype_input)     
        self.skips = nn.ModuleList([skip_setup() for _ in range(spconv_layers)])    
        # zero pad level         
        self.num_pad, self.num_unpad = self._get_pad_unpad_levels(fourier_pad_level, fft_dim=fft_dim)
        # nonlinear activation        
        self.act = non_linearity 
        # status 
        print(f'Initialize {fft_dim:d}D-FNO model for complex function ... \n')


    def _get_pad_unpad_levels(self, pad_level: int | tuple[int] | list[int], 
                              fft_dim: int) -> tuple[tuple[int], tuple[int]]:   
        if pad_level is None:
            pad_level = 0
        elif isinstance(pad_level, int):
            pad_level = [pad_level]
        else:
            pad_level = list(pad_level)
        # 
        if len(pad_level) < fft_dim:
            pad_level += [0]*(fft_dim - len(pad_level))
        elif len(pad_level) > fft_dim:
            pad_level = pad_level[:fft_dim]  
        # pad level
        num_pad = []
        for i in reversed(pad_level): num_pad += [0, i]
        num_pad = tuple(num_pad)
        # unpad level
        num_unpad = list()
        for p in pad_level:
            if p > 0:
                num_unpad.append(slice(None, -p, None))
            else:
                # pad=0, None
                num_unpad.append(slice(None, None, None))
        num_unpad = (Ellipsis,) + tuple(num_unpad)
        return num_pad, num_unpad
    
    def forward(self, x):
        '''
        x: (batch, in, ...) -> (batch, out, ...) 
        '''          
        length_fourier = len(self.spconvs)
        # zero padding parameters
        num_pad, num_unpad = self.num_pad, self.num_unpad
        # add zero padding           
        if max(num_pad) > 0: x = F.pad(x, num_pad, 'constant', 0)   
        # spconv, skip 
        for i in range(length_fourier):
            x1 = self.spconvs[i](x)
            x2 = self.skips[i](x)
            x = x1 + x2             
            # act           
            if i < length_fourier - 1:
                x = self.act(x)               
        # remove zero padding
        if max(num_pad) > 0: x = x[num_unpad]                
        # return
        return x
    



