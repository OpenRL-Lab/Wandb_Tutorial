sweep_config = {
    "name": "sweep_with_launchpad",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "grid",
    "parameters": {
        "a": {
            "values": [1, 2, 3, 4]
        }
    }
}