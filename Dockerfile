FROM python:3.10

#Set working directory inside container
WORKDIR /code

#Copy requirements file
COPY ./requirements.txt /code/requirements.txt

# Install system libraries + Python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /code/requirements.txt

#Copy the rest of the project into the container
COPY . /code

#Expose FastAPI Port
EXPOSE 8000

#Start FastAPI with uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0","--port", "8000"]