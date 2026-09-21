from lightgbm import LGBMRegressor

from src.config import DEFAULT_LIGHTGBM_PARAMS

def create_lightgbm_model(
    params: dict | None = None,
) -> LGBMRegressor:
    model_params = DEFAULT_LIGHTGBM_PARAMS.copy()

    if params is not None:
        model_params.update(params)

    return LGBMRegressor(**model_params)

def train_model(
    model: LGBMRegressor,
    X_train,
    y_train,
) -> LGBMRegressor:
    model.fit(X_train, y_train)

    return model

def predict_model(
    model: LGBMRegressor,
    X,
):
    return model.predict(X)