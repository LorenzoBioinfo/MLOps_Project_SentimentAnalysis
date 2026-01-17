from prefect import flow, task
from prefect_docker import DockerContainer


COMMON_VOLUMES = {
    "./data": "/app/data",
    "./models": "/app/models",
    "./reports": "/app/reports",
    "./huggingface_cache": "/app/huggingface_cache",
}

COMMON_ENV = {
    "HF_HOME": "/app/huggingface_cache",
    "BASE_DIR": "/app",
}


@task(
    name="Data preparation",
    retries=2,
    retry_delay_seconds=60,
    timeout_seconds=1800,  
)
def data_prep():
    DockerContainer(
        image="sentiment-data-prep",
        command=[
            "sh", "-c",
            "python src/download_data.py /app/data/raw && "
            "python src/data_preparation.py tweet_eval --output_dir /app/data/processed"
        ],
        volumes=COMMON_VOLUMES,
        env=COMMON_ENV,
        auto_remove=True,
    ).run()


@task(
    name="Monitoring",
    retries=1,
    timeout_seconds=900,
)
def monitoring():
    DockerContainer(
        image="sentiment-monitoring",
        command=["python", "src/monitoring.py"],
        volumes=COMMON_VOLUMES,
        env=COMMON_ENV,
        auto_remove=True,
    ).run()


@task(
    name="Inference App",
    retries=0,
)
def app():
    DockerContainer(
        image="sentiment-app",
        command=[
            "uvicorn", "src.app:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ],
        volumes=COMMON_VOLUMES,
        env=COMMON_ENV,
        ports={8000: 8000},
        auto_remove=False,  
    ).run()


@flow(
    name="sentiment-analysis-pipeline",
    log_prints=True,
)
def sentiment_pipeline(run_app: bool = True):
    data_prep()
    monitoring()
    if run_app:
        app()


if __name__ == "__main__":
    sentiment_pipeline()
