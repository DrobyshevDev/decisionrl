# Serve an exported decisionrl policy over HTTP.
#
# Build (from the repo root, with deploy/models/policy.onnx + policy.onnx.json present):
#   docker build -f deploy/Dockerfile -t decisionrl-serve .
# Run:
#   docker run -p 8000:8000 decisionrl-serve
#   curl localhost:8000/health
#   curl -X POST localhost:8000/predict -H 'content-type: application/json' \
#        -d '{"observation": [0, 0, 0, 0]}'
#
# The serving image needs neither PyTorch nor the decisionrl package, only onnxruntime +
# FastAPI, so it stays small. The entrypoint is the standalone deploy/serve.py.
FROM python:3.14-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2
# Pinned by digest, not by tag. `python:3.14-slim` is republished whenever the
# base is rebuilt, so the image this Dockerfile produced last week and the one
# it produces today are not the same image and nothing records which was which.
# Dependabot raises the digest and the tag comment together.

WORKDIR /app

RUN pip install --no-cache-dir onnxruntime "fastapi>=0.100" "uvicorn[standard]>=0.23" numpy

# The exported policy (override DECISIONRL_MODEL to point elsewhere at runtime).
COPY deploy/models /models
COPY deploy/serve.py .
ENV DECISIONRL_MODEL=/models/policy.onnx

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
