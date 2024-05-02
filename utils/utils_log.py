import inspect
import torch
### PRINT ARGUMENTS ON TERMINAL
def print_args(func):
    def wrapper(*args, **kwargs):
        argspec = inspect.getfullargspec(func)
        print(f"Class '{func.__qualname__}' arguments: {argspec.args}")

        def print_tensor(name, shape):
            print(f"    Tensor: {name}, Shape: {shape}")
        
        def print_div(num):
            print(f"_"*num)

        for argname, argval in zip(argspec.args, args):
            if isinstance(argval, torch.Tensor):
                print_div(20)
                print_tensor(argname, argval.shape)
                
            if isinstance(argval, list) and all(isinstance(tensor, torch.Tensor) for tensor in argval):
                print_div(20)
                for idx, tensor in enumerate(argval):
                    print_tensor(argname + str(idx), tensor.shape)
                
        return func(*args, **kwargs)
    return wrapper


import logging
logging.basicConfig(level=logging.INFO)
### LOG ARGUMENTS TO FILE
def log_args(func):
    def wrapper(*args, **kwargs):
        argspec = inspect.getfullargspec(func)
        arg_dict = {k: v for k, v in zip(argspec.args, args)}
        arg_dict.update(kwargs)
        logging.info(f"Function '{func.__name__}' arguments and values: {arg_dict}")
        return func(*args, **kwargs)
    return wrapper


def interactive_warnings(msg : str):
    warning_length = len(msg)
    padding = 100
    half_padding = padding // 2
    separate_line = "=" * (warning_length + padding)
    
    print(f'{separate_line}\n' * 3 + " " * half_padding + f'{msg}\n' + f'{separate_line}\n' * 2)
    #input("Press enter to continue...")

def get_class_path(cls):
    class_hierarchy = [cls.__name__ for cls in cls.__mro__]
    class_hierarchy.remove('object')
    class_hierarchy.remove('Module')
    class_hierarchy.reverse() # returns None and mod inplace
    return '/'.join(class_hierarchy)