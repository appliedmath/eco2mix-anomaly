# Airflow 2.11 (same Airflow 2 syntax as the course) + the Docker provider,
# so DAG tasks can start our existing trainer image (DockerOperator).
FROM apache/airflow:2.11.2-python3.12

RUN pip install --no-cache-dir \
    "apache-airflow==2.11.2" \
    "apache-airflow-providers-docker==4.5.2"
