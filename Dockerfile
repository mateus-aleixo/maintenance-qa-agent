# One image, two habitats: a normal web server locally (docker compose up) and
# unchanged on AWS Lambda -- the Lambda Web Adapter extension turns function
# invocations into plain HTTP. Same arrangement as conformal-rul and conformal-seg.
# The adapter is its own stage so CI can build against a stand-in: anonymous pulls
# from ECR Public fail on shared runners ("toomanyrequests: Data limit exceeded").
ARG LAMBDA_ADAPTER_IMAGE=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1
FROM ${LAMBDA_ADAPTER_IMAGE} AS lambda-adapter

FROM python:3.12-slim

COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

COPY requirements/serve.txt requirements/serve.txt
RUN pip install --no-cache-dir -r requirements/serve.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# --no-deps: the pinned serving set above is the whole environment. A normal
# install would pull pypdf, and the embed extra would pull sentence-transformers.
RUN pip install --no-cache-dir --no-deps .

# Serving registry: ONNX query encoder + tokenizer + the corpus index + the one
# calibrated gate. Built locally with `python -m conformal_rag.registry` and
# `python scripts/export_embedder.py`; not committed (135 MB of encoder).
COPY models ./models

ENV MODEL_ROOT=/app/models \
    PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health

EXPOSE 8000
CMD ["uvicorn", "conformal_rag.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
