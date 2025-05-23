import pickle
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.random_forest_model import get_model as RandomForestRegressor

def train_and_evaluate(train_path, test_path, model_dir, model_filename, n_estimators, seed):
    # Load data
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    print(f"Test R^2 score: {score:.4f}")

    # Create output directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    model_filepath = os.path.join(model_dir, model_filename)

    # Save model
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else 'params.yaml'

    with open(params_path) as f:
        params = yaml.safe_load(f)

    train_path = os.path.join(params['prepare']['out_dir'], 'train.pkl')
    test_path = os.path.join(params['prepare']['out_dir'], 'test.pkl')

    model_dir = params['train']['model_out']  # This is the directory path
    model_filename = 'rf_model.pkl'  # Hardcoded or could be param if you want

    n_estimators = params['train']['n_estimators']
    seed = params['train']['seed']

    train_and_evaluate(train_path, test_path, model_dir, model_filename, n_estimators, seed)