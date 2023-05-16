import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import used_cars


def run_rfe(X_train, X_test, y_train, y_test, n_features_to_select):
    rfe = RFE(
        estimator=RandomForestRegressor(random_state=0),
        n_features_to_select=n_features_to_select,
        step=1,
        verbose=1
    )
    rfe.fit(X_train, y_train)
    y_pred = rfe.predict(X_test)
    return (n_features_to_select, r2_score(y_test, y_pred), rfe)


if __name__ == "__main__":
    usr_in = input("Do used_cars.remove_null_rows()? (Y/n): ")

    df = pd.read_feather(
        "dataset/used_cars_data_medium.feather", used_cars.Info.columns
    ).fillna(pd.NA)

    if usr_in != "n":
        df = used_cars.remove_null_rows(df, [
            'back_legroom', 'front_legroom', 'fuel_tank_volume', 'height', 'length',
            'maximum_seating', 'width', 'body_type', 'fuel_type', 'transmission',
            'wheel_system', 'engine_type', 'power', 'torque'
        ])

    df.insert(df.shape[1]-1, "price", df.pop("price"))

    X = pd.DataFrame(df.iloc[:, :-1])
    y = pd.DataFrame(df.iloc[:, -1])["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    transformer = used_cars.make_used_cars_transformer(random_state=0)
    feat_selector = used_cars.FinalFeatureSelector()

    Xt_train = transformer.fit_transform(X_train, y_train)
    Xt_train = feat_selector.fit_transform(Xt_train)
    Xt_test = transformer.transform(X_test)
    Xt_test = feat_selector.transform(Xt_test)

    rfe_results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(run_rfe)(Xt_train, Xt_test, y_train, y_test, i)
        for i in range(1, len(Xt_train.columns)+1, 1)
    )

    if usr_in != "n":
        filename = "rfe_results_36_medium"
    else:
        filename = "rfe_results_36_medium_r"

    with open(f"{filename}.joblib", "wb") as f:
        joblib.dump(rfe_results, f)
