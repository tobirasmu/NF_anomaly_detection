from pathlib import PosixPath, Path
import pickle


def save_pickle(obj: any, path: PosixPath, verbose=True):
    path = Path(path)
    with open(path.with_suffix('.pkl'), 'wb') as file:
        pickle.dump(obj, file)
    if verbose:
        print(f"Object pickled to {str(path)}")
