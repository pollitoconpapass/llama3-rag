FROM python:3.12.2-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "app.py"]
