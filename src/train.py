import pickle
import yaml
import sys
import os

# import from models module
from models.random_forest_model import get_model

def train_and_evaluate(train_path, test_path, model_out, n_estimators, seed):
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    model = get_model(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Test R^2 score: {score:.4f}")

    model_filepath = os.path.join(model_out, "rf_model.pkl")
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Saved trained model to {model_out}")

if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else 'params.yaml'
    with open(params_path) as f:
        params = yaml.safe_load(f)

    train_path = params['prepare']['out_dir'] + '/train.pkl'
    test_path = params['prepare']['out_dir'] + '/test.pkl'
    model_out = params['train']['model_out']
    n_estimators = params['train']['n_estimators']
    seed = params['train']['seed']

    train_and_evaluate(train_path, test_path, model_out, n_estimators, seed)