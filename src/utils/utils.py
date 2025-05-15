import os 
from contextlib import contextmanager

@contextmanager
def rank0_context():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    try:
        if local_rank == 0:
            yield
        else:
            yield
    finally:
        pass