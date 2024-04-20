import fire
import sys
import os
from torch import cuda


def print_available_gpus(total=None, n=None, min_usage=1,
    print_used_count=False, print_to_file='/dev/stdout'):
    total = cuda.device_count() if total is None else total
    all_gpus = [cuda.device(i) for i in range(total)]
    avail_gpus = []
    for i, gpu in enumerate(all_gpus):
        info = cuda.mem_get_info(device=gpu)
        used_GiB = (info[1] - info[0]) / (1024 * 1024 * 1024)
        if used_GiB < min_usage:
            avail_gpus.append(str(i))
    if n is None: n = total
    result = avail_gpus[:n]
    with open(print_to_file, 'w') as fh:
        if print_used_count:
            print(total - len(result), file=fh)
        else:
            print(','.join(result), file=fh)


if __name__ == '__main__':
    os.environ["PAGER"] = "cat"
    fire.Fire(print_available_gpus)
