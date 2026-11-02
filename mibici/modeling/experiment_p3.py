import mlflow

from mibici.config import MLFLOW_EXPERIMENT_NAME
from mibici.modeling.train_p3 import main as train_p3


def main():

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    configurations = [
        {"n_clusters": 4, "n_init": 10, "max_iter": 300},
        {"n_clusters": 5, "n_init": 10, "max_iter": 300},
        {"n_clusters": 6, "n_init": 10, "max_iter": 300},
    ]

    with mlflow.start_run(
        run_name="P3_busqueda_KMeans_real"
    ):

        mlflow.set_tags({
            "team": "46",
            "question": "P3",
            "search_type": "grid_search",
            "stage_of_work": "candidate",
        })

        for params in configurations:

            print(
                f"Entrenando KMeans con "
                f"K={params['n_clusters']}"
            )

            train_p3(
                model_name="kmeans",
                hyperparameters=params
            )

    print("Búsqueda de P3 finalizada.")


if __name__ == "__main__":
    main()