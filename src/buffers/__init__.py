from .reservoir_buffer import ReservoirBuffer

def get_buffer(buffer_type: str,
               mem_size: int = 2000,
               alpha_ema: int = 0.5,
               device: str = 'cpu'):
    if buffer_type == 'reservoir':
        return ReservoirBuffer(mem_size, alpha_ema, device=device)
    else:
        raise Exception(f'Buffer type {buffer_type} is not supported')