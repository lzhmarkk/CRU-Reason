from tqdm.autonotebook import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def multiprocess_call_list(fn, samples, cached=None, n_workers=8, **kwargs):
    if cached is None: cached = []

    while len(cached) < len(samples):
        cached.append(None)

    assert len(samples) == len(cached)

    futures = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for key, sample in enumerate(samples):
            if cached[key] is None:
                future = executor.submit(fn, *(key, sample), **kwargs)
                futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            key, result = future.result()
            cached[key] = result

    return cached


def multiprocess_call_dict(fn, samples, cached=None, n_workers=8, **kwargs):
    if cached is None: cached = {}

    futures = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for key, sample in samples.items():
            if key not in cached or cached[key] is None:
                future = executor.submit(fn, *(key, sample), **kwargs)
                futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            key, result = future.result()
            cached[key] = result

    return cached


def multiprocess_call(fn, samples, cached=None, n_workers=8, **kwargs):
    if isinstance(samples, list):
        return multiprocess_call_list(fn, samples, cached, n_workers, **kwargs)
    elif isinstance(samples, dict):
        return multiprocess_call_dict(fn, samples, cached, n_workers, **kwargs)
    else:
        raise TypeError('samples and cached must be of same type')
