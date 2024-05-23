import os
import fire
import json
from torch import cuda


def priority_Q(devices):
    GiB = 1024 * 1024 * 1024
    gpus = [cuda.device(i) for i in range(cuda.device_count())]
    def mapper(ordinal, info):
        avail = round(info[0] / GiB, 3)
        total = round(info[1] / GiB, 3)
        occupied = sum(devices[ordinal].values()) if ordinal in devices else 0
        return round(avail - occupied, 3), total
    info = [
        (i, *mapper(str(i), cuda.mem_get_info(device=gpu)))
        for i, gpu in enumerate(gpus)
    ]
    return sorted(info, key=lambda x: x[1], reverse=True)


def allocate(runid, budget, db_file='./gpu_assigner_db.json',
             min_vram=1, verbose=False, output_file='/dev/stdout'):
    open(output_file, 'w').close()
    assert os.path.exists(db_file)
    with open(db_file, 'r') as fh:
        db = json.load(fh)
    devices, runs = db['devices'], db['runs']
    assert runid not in runs

    n, vram_budget = budget.split('x')
    n, vram_budget = int(n), float(vram_budget)
    if verbose:
        print('allocating for', runid, f'n_gpu={n}, budget={vram_budget}')

    inf = float('inf')
    allocated_devices = []
    for _ in range(n):
        for i, avail, total in priority_Q(devices):
            if verbose:
                print(f'GPU#{i}: {avail:.3f} / {total:.3f} GiB available')
            if (avail > vram_budget or vram_budget == inf) and avail > min_vram:
                ordinal = str(i)
                if verbose: print('allocated device', ordinal)
                allocated_devices.append(ordinal)
                if ordinal not in devices: devices[ordinal] = dict()
                devices[ordinal][runid] = vram_budget
                if runid not in runs: runs[runid] = dict()
                runs[runid][ordinal] = vram_budget
                break
        else:
            if verbose: print('no available device!')
            break
    else:
        if verbose:
            print('allocation successful!')
            print(json.dumps(db, indent=2))
        with open(db_file, 'w') as fh:
            json.dump(db, fh)
        with open(output_file, 'w') as fh:
            fh.write(','.join(allocated_devices))


def refresh(*all_runids, db_file='./gpu_assigner_db.json'):
    if len(all_runids) == 0 or not os.path.exists(db_file):
        db = {'devices': {}, 'runs': {}}
    else:
        with open(db_file, 'r') as fh:
            db = json.load(fh)

    dead_runids = set(db['runs'].keys()) - set(all_runids)
    for runid in dead_runids:
        print('remove', runid)
        runid_devices = db['runs'][runid]
        for dev_key in runid_devices.keys():
            del db['devices'][dev_key][runid]
        del db['runs'][runid]

    print(json.dumps(db, indent=2))
    with open(db_file, 'w') as fh:
        json.dump(db, fh)


if __name__ == '__main__':
    os.environ["PAGER"] = "cat"
    fire.Fire(dict(
        allocate=allocate,
        refresh=refresh
    ))
