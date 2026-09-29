# FROM python:3.12-slim

# WORKDIR /app

# COPY requirements.txt ./
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# COPY loan_approval_dataset.csv train_model.py ./

# RUN mkdir -p /app/output

# CMD python train_model.py && cp loan_model.pkl /app/output/

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]