# Operator app (M4). It reads nothing from disk: every number comes from the API.
FROM python:3.12-slim
WORKDIR /app
COPY docker/requirements-streamlit.txt .
RUN pip install --no-cache-dir -r requirements-streamlit.txt
COPY streamlit_app/ streamlit_app/
RUN useradd --create-home app
USER app
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
