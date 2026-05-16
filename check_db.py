import mlflow

mlflow.set_tracking_uri("file:./mlruns")
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"ID: {exp.experiment_id} | Name: {exp.name}")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    print(f"   -> Found {len(runs)} runs in this experiment.")