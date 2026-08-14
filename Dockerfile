# light python image
FROM python:3.11-slim

WORKDIR /app

# install dependencies into the working directory
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the full code to the container
COPY . .

# run main flow for the moment
CMD ["python", "main.py"]