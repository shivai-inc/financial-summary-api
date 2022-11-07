FROM python:3.10

WORKDIR /api

COPY requirements.txt ./

RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000:8000

CMD ["uvicorn", "api:summary_api", "--host", "0.0.0.0", "--port", "8000"]










