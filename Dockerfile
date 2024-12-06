FROM public.ecr.aws/lambda/python:3.12

RUN dnf -y install \
    libGL \
    libGLU \
    && dnf clean all

COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

COPY saved_model /opt/model/saved_model

COPY index.py .

CMD ["index.lambda_handler"]