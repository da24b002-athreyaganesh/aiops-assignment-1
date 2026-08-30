import mlflow
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


X, y = fetch_openml('mnist_784', version = 1, return_X_y=True, as_frame=False)
X = X/255.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=42)


mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('mnist-mlp-athreya')

experiments = [
    {'learning_rate':0.0001, 'hidden_layers':(128,)},
    {'learning_rate':0.0001, 'hidden_layers':(256, 128)},
    {'learning_rate':0.001, 'hidden_layers':(128,)},
    {'learning_rate':0.001, 'hidden_layers':(256, 128)},
    {'learning_rate':0.01, 'hidden_layers':(128,)},
    {'learning_rate':0.01, 'hidden_layers':(256, 128)}
]


for idx, hp in enumerate(experiments, start = 1):
    with mlflow.start_run(run_name = f"mlp_run_{idx}"):
        clf = MLPClassifier(
            hidden_layer_sizes = hp['hidden_layers'],
            learning_rate_init = hp['learning_rate'],
            max_iter = 20,
            random_state = 42
        )
        clf.fit(X_train, y_train)

        train_loss = clf.loss_
        val_acc = clf.score(X_val, y_val)

        mlflow.log_param("learning_rate_init", hp['learning_rate'])
        mlflow.log_param("hidden_layer_sizes", hp['hidden_layers'])
        mlflow.log_param("max_iter", 20) # parameter tracking

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("val_accuracy", val_acc) #metric tracking

# all runs are over now
#kill the server by closing the terminal :)