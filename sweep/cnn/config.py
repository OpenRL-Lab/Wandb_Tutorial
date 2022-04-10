hyperparameter_defaults = dict(
    dropout = 0.5,
    channels_one = 16,
    channels_two = 32,
    batch_size = 100,
    learning_rate = 0.001,
    epochs = 2,
    )

sweep_config = {
    "name": "sweep_with_launchpad_cnn",
    "metric": {"name": "loss", "goal": "minimize"},
    "method": "grid",
    "parameters": {
        "dropout": {
            "values": [0.1, 0.3, 0.6, 0.9]
        }
    }
}