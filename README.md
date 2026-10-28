# Diabetes Prediction API

An end-to-end machine learning project: a model is trained in **Google Colab**
on a **Kaggle** dataset, saved to disk, and served as a **FastAPI** REST API that
is packaged with **Docker** and validated by a **GitHub Actions** CI pipeline.

## Dataset

- **Source:** [Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/salihacur/diabetes)
- **File:** `diabetes.csv` (768 rows, 8 features, 1 target)
- **Target:** `Outcome` — `0` = no diabetes, `1` = diabetes

The dataset is downloaded from Kaggle and stored in the `data/` folder.

## Project workflow

1. **Get the data** — download `diabetes.csv` from the Kaggle dataset page above.
2. **Train the model in Colab** — open `notebook/Diabetes_Model_Training.ipynb`
   in Google Colab, upload `diabetes.csv` to the Colab session, and run all the
   cells. The notebook:
   - explores the data,
   - splits it into train / test sets,
   - trains a scikit-learn `Pipeline` (impute → scale → RandomForest),
   - evaluates it on the test set,
   - saves the model to `model/model.joblib` and the results to `model/metrics.json`,
   - reloads the saved model and makes a test prediction.
3. **Serve the model** — the FastAPI app in `app/` loads `model.joblib` and
   exposes prediction endpoints.
4. **Package with Docker** — the `Dockerfile` builds a container that runs the API.
5. **Automate with CI** — `.github/workflows/ci.yml` runs the tests and builds
   the Docker image on every push and pull request.

## Project structure

```
.
├── notebook/
│   ├── Diabetes_Model_Training.ipynb          # training notebook (run in Colab)
│   └── Diabetes_Model_Training_output.ipynb   # executed notebook with results
├── model/
│   ├── model.joblib                           # trained model
│   └── metrics.json                           # model parameters and accuracy
├── app/
│   ├── __init__.py
│   ├── main.py                                # FastAPI application
│   └── schemas.py                             # request / response models
├── tests/
│   └── test_api.py                            # automated API tests
├── data/
│   └── diabetes.csv                           # dataset downloaded from Kaggle
├── Dockerfile
├── .dockerignore
├── .gitignore
├── conftest.py                                # puts the project root on the path for tests
├── requirements.txt
└── .github/workflows/ci.yml
```

## Model

- **Algorithm:** Random Forest classifier (200 trees)
- **Preprocessing:** median imputation + standard scaling, bundled in a pipeline
- **Test accuracy:** see `model/metrics.json`

## Running the API locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs for the interactive API documentation.

## Running with Docker

```bash
docker build -t diabetes-ml-api .
docker run -p 8000:8000 diabetes-ml-api
```

## API endpoints

| Method | Path       | Description                           |
|--------|------------|---------------------------------------|
| GET    | `/`        | Welcome message                       |
| GET    | `/health`  | Health check                          |
| POST   | `/predict` | Predict diabetes for one patient      |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":6,"Glucose":148.0,"BloodPressure":72.0,"SkinThickness":35.0,"Insulin":0.0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50.0}'
```

### Example response

```json
{"prediction": 1, "label": "Diabetes", "probability": 0.67}
```

## Tests

```bash
python -m pytest -v
```

## CI

Every push or pull request to `main` triggers the GitHub Actions workflow, which
installs the dependencies, runs the tests with `pytest`, and builds the Docker
image to confirm the application packages correctly.