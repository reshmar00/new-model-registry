import h5py
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import yaml
import sys

def load_data(h5_filepath):
    with h5py.File(h5_filepath, 'r') as f:
        group = f['data']['1']['meshes']
        x = group['x'][:]
        y = group['y'][:]
        z = group['z'][:]
        B = group['B'][:]

        X = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        y_data = B.reshape(-1, B.shape[-1])

    return X, y_data

def split_and_save(X, y, test_size, random_state, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    train_path = os.path.join(out_dir, 'train.pkl')
    test_path = os.path.join(out_dir, 'test.pkl')

    with open(train_path, 'wb') as f:
        pickle.dump({'X': X_train, 'y': y_train}, f)
    with open(test_path, 'wb') as f:
        pickle.dump({'X': X_test, 'y': y_test}, f)

    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")

if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else 'params.yaml'
    with open(params_path) as f:
        params = yaml.safe_load(f)

    h5_filepath = params['prepare']['h5_filepath']
    test_size = params['prepare']['test_size']
    random_state = params['prepare']['random_state']
    out_dir = params['prepare']['out_dir']

    X, y = load_data(h5_filepath)
    split_and_save(X, y, test_size, random_state, out_dir)