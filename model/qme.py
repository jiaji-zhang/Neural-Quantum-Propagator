import math, warnings
import torch
import torch.utils.data

from typing import Callable


    
# General class of QME, using csr sparse tensor
class QME:
    """General class for Quantum Master Equation

    Arguments
    ----------
    dim_time: int
        Number of time points
    dim_sys: int
        Number of entries in density matrix (total)
    file_L_matrix: str
        File (.pt) for L_matrix (right-hand side of QME)
    device: ...
    """
    def __init__(self, dim_time: int, dim_sys: int, file_L_matrix: str,
                 device: torch.device = 'cpu') -> None:        
        self.dim_time = dim_time
        self.dim_sys = dim_sys     
        self.dtype = torch.cfloat
        self.device = device           
        # matrix for right-hand side of PDE
        if not file_L_matrix.endswith('.pt'): file_L_matrix += '.pt'
        warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
        self.L_matrix = torch.load(file_L_matrix).to(self.device)            
        print('Load matrix for right-hand side of QME ... \n')       
        # t_interval, in general [0,1)
        self.t_interval = 1.
        

    # Right-hand sided of QME
    def _compute_pde(self, L_matrix: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        rho_input_size = rho.shape
        N_liou = rho.shape[-1]
        rho = rho.reshape(-1, N_liou)
        res = torch.zeros(rho.shape, dtype=rho.dtype, device=rho.device)
        for s in range(rho.shape[0]):
            res[s, :] = L_matrix.matmul(rho[s, :]).reshape(N_liou)
        res = res.reshape(rho_input_size)
        return res
    
    # prototype rk4 solver of one-step
    def _rk4_one_step(self, dt: float, rho_t: torch.Tensor) -> torch.Tensor:
        sub1 = self._compute_pde(self.L_matrix, rho_t)
        sub2 = self._compute_pde(self.L_matrix, rho_t + 0.5 * dt * sub1)
        sub3 = self._compute_pde(self.L_matrix, rho_t + 0.5 * dt * sub2)
        sub4 = self._compute_pde(self.L_matrix, rho_t + dt * sub3)
        rho_t += dt * (sub1 + 2. * sub2 + 2. * sub3 + sub4) / 6.
        return rho_t
    
    def RK4_solver(self, rho_init: torch.Tensor, dt: float, steps: int) -> torch.Tensor:
        """RK4 solver 

        Arguments
        ----------
        rho_init: tensor
            initial condition
        dt: float
            time step
        steps: int
            steps to be propagated        
        """
        for _ in range(steps): 
            rho_init = self._rk4_one_step(dt, rho_init)
        return rho_init


    
    # Residual (L_phys) of QME
    def PDE_residual(self, rho_out: torch.Tensor) -> torch.Tensor:   
        Nb = rho_out.shape[0]
        Nt = rho_out.shape[-1]
        rho_out = rho_out.reshape(Nb, -1, Nt)
        # permute (b,x,t) -> (b,t,x)
        rho_out = rho_out.permute(0, 2, 1)
        rhs_L_rho = self._compute_pde(self.L_matrix, rho_out) 
        # permute (b,t,x) -> (b,x,t)
        rhs_L_rho = rhs_L_rho.permute(0, 2, 1)
        return rhs_L_rho 


    # extract population
    def Extract_population(self, rho_time: torch.Tensor) -> torch.Tensor:
        Ns = int(math.sqrt(self.dim_sys))
        # reshape to (b,s,s,t)
        Nb = rho_time.shape[0]
        Nt = rho_time.shape[-1]
        rho_time = rho_time.reshape(Nb, Ns, Ns, Nt)
        # permute (b,s,s,t) -> (b,t,s,s)
        rho_time = rho_time.permute(0, 3, 1, 2)
        # extract diagonal (b,t,s,s) -> (b,t,s,s)
        rho_diag = rho_time.diagonal(0, -1, -2)
        # permute (b,t,s) -> (b,s,t)
        rho_diag = rho_diag.permute(0, 2, 1)
        return rho_diag 
    
    # load time evolution / response function data
    def Load_data_file(self, file_name: str | list[str], sample_size: int, offset: int = 0) -> torch.Tensor:
        '''
        Shape is always (sample, self.dim_data)
        '''
        if isinstance(file_name, str):
            if not file_name.endswith('.pt'): file_name += '.pt'
            data_read = torch.load(file_name)
        elif isinstance(file_name, list) or isinstance(file_name, tuple):
            data_read = []
            for fx in list(file_name):
                if not fx.endswith('.pt'): fx += '.pt'
                data_read.append(torch.load(fx))
            data_read = torch.cat(data_read, dim=0)
        # reshape
        data_read = data_read[offset : offset + sample_size, ...]
        return data_read.to(self.device) 
    


# Physical operator
class Physop:
    """Class for physical operator (used in response function calculation)

    Arguments
    ----------
    data_vec: tensor
        Data of operator
    device: ...   
    """
    def __init__(self, data_vec: torch.Tensor, device: torch.device = 0) -> None:
        if len(data_vec.shape)!= 2 and data_vec.shape[0] != data_vec.shape[1]:
            raise ValueError(f'Physical operator, only support square matrix.')
        self.data = data_vec.to(device)
     
    # apply physical operator to vectorize-function
    def Apply_to(self, rho_vec: torch.Tensor) -> torch.Tensor:       
        rho_shape = rho_vec.shape
        Nx = self.data.shape[0]        
        # physop(Nx, Nx) x rho_vec(..., Nx) 
        rho_vec = rho_vec.reshape(-1, Nx)
        for n in range(rho_vec.shape[0]):
            rho_vec[n, :] = self.data.matmul(rho_vec[n, :])              
        rho_vec = rho_vec.reshape(rho_shape)
        return rho_vec    

    # Trace, x should be the last dimension
    def Trace_last_dim(self, rho_vec: torch.Tensor) -> torch.Tensor:
        '''
        rho_vec: (batch, time, x)
        '''     
        rho_vec = self.Apply_to(rho_vec)
        Ns = int(math.sqrt(self.data.shape[0])) 
        rho_vec = rho_vec.reshape(*rho_vec.shape[:-1], Ns, Ns).diagonal(0, -1, -2)   
        rho_vec = rho_vec.sum(-1) # if is_trace else rho_vec
        return rho_vec



# Gaussian unitary ensemble
def _random_matrix(length: int, device: torch.device = 'cpu') -> torch.Tensor:
    """Gaussian unitary ensemble (Hermitian random matrix)

    Arguments
    ----------
    length: int
        row/col of matrix
    device: ...    
    """
    res = torch.randn(length * length, dtype=torch.cfloat, device=device).reshape(length, length)
    res = (res + torch.conj_physical(res.transpose(0, 1))) / 2
    res = res / torch.max(torch.abs(res))
    res *= 0.999
    return res
    


# random initial conditions
def Online_sample_qme(dims_other: int | tuple[int], dim_system: int, 
                      device: torch.device = 0) -> torch.Tensor:
    """Random initial conditions sampler for physics-informed loss function

    Arguments
    ----------
    dims_other: tuple[int]
        batch size
    dim_system: int
        total length (row*col) of density matrix
    device: ...
    
    """
    '''
    Reduced density matrix: Gaussian Unitary Ensemble (GUE).
    Return: tensor (..., system)
    '''
    length = int(math.sqrt(dim_system))
    if isinstance(dims_other, int): dims_other = tuple([dims_other])
    res = [_random_matrix(length, device) for _ in range(math.prod(dims_other))]
    res = torch.cat(res, dim=0).reshape(*dims_other, dim_system)
    return res



