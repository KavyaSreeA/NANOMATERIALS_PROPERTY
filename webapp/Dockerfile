# Build:  docker build -f webapp/Dockerfile -t stiffness-app .   (run from the repository root)
# Run:    docker run -p 8000:8000 stiffness-app
FROM node:20-slim AS frontend
WORKDIR /fe
COPY webapp/frontend/package*.json ./
RUN npm install
COPY webapp/frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /srv
# bibtexparser<2 needs an old setuptools to build (pulled in by matminer)
RUN pip install --no-cache-dir "setuptools<70" wheel
RUN pip install --no-cache-dir --no-build-isolation "bibtexparser<2"
COPY webapp/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY webapp/backend/app ./app
COPY mat-ml/models ./models
COPY --from=frontend /fe/dist ./frontend_dist
ENV MODELS_DIR=/srv/models FRONTEND_DIST=/srv/frontend_dist
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
