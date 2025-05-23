import h5py
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import yaml
import sys

def load_data(h5_filepath):
    with h5py.File(h5_filepath, 'r') as f:
        mesh_group = f['data']['1']['meshes']

        # Access the datasets under the 'B' group
        x = mesh_group['B']['x'][:]
        y = mesh_group['B']['y'][:]
        z = mesh_group['B']['z'][:]

        # Use x, y, z to construct feature and target arrays
        X = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        y_data = np.stack([x, y, z], axis=-1).reshape(-1, 3)

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

# import h5py

# h5_filepath = "data/example-femm-3d.h5"

# def explore_h5(filepath):
#     def recursive_print(name, obj):
#         print(name)

#     with h5py.File(filepath, "r") as f:
#         print("Top-level keys:", list(f.keys()))
#         print("\nFull structure:")
#         f.visititems(lambda name, obj: print(f"{name} ({type(obj)})"))

# explore_h5(h5_filepath)