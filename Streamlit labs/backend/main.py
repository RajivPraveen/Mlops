from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data, predict_proba

app = FastAPI(title="Wine Classification API")


class WineFeatures(BaseModel):
    """Input features for wine classification (13 chemical measurements)."""
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float


class WineResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict


WINE_CLASSES = {0: "Class 0 (Cultivar 1)", 1: "Class 1 (Cultivar 2)", 2: "Class 2 (Cultivar 3)"}


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.post("/predict", response_model=WineResponse)
async def predict_wine(features: WineFeatures):
    """
    Predict the wine cultivar class based on 13 chemical measurements.

    Returns the predicted class, class name, and probability distribution.
    """
    try:
        feature_values = [[
            features.alcohol, features.malic_acid, features.ash,
            features.alcalinity_of_ash, features.magnesium, features.total_phenols,
            features.flavanoids, features.nonflavanoid_phenols,
            features.proanthocyanins, features.color_intensity,
            features.hue, features.od280_od315, features.proline
        ]]

        prediction = predict_data(feature_values)
        probabilities = predict_proba(feature_values)

        pred_class = int(prediction[0])
        prob_dict = {
            WINE_CLASSES[i]: round(float(p), 4)
            for i, p in enumerate(probabilities[0])
        }

        return WineResponse(
            prediction=pred_class,
            class_name=WINE_CLASSES[pred_class],
            probabilities=prob_dict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
