import functools  
import random, numpy as np
import torch
def seed_worker(worker_id, base_seed):  
    worker_seed = base_seed + worker_id  
    random.seed(worker_seed)  
    np.random.seed(worker_seed)  
    torch.manual_seed(worker_seed)

    
def set_seed(seed=42):  
    import random, numpy as np
    # 设置全局种子  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True  
    torch.use_deterministic_algorithms(True, warn_only=True)
    # 返回部分应用函数，固定base_seed  
    return functools.partial(seed_worker, base_seed=seed)
