"""
SC1015 Mini Project
===================

Dataset: https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset
"""

import re
import time

import joblib
import numpy as np
import pandas as pd
from pyarrow import feather
from sklearn import base
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_validate


class Info:
    """
    MODIFY THIS CLASS INSTEAD OF PASSING ARGUMENTS TO THE CUSTOM
    TRANSFORMERS 

    Information about dataset features/columns and their data types and
    default values used by the custom transformers. This class is
    inherited by all custom transformers. Values of the attributes can
    be changed. For example, ::

        BaseInfo.null_columns = ["bed", "engine_type"]

    Attributes
    ----------
    columns : list of str
        All columns from the dataset.

    columns_dtype : dict
        All columns from the dataset and their data types.

    null_columns : list of str
        All columns with ~100% null values.

    useless_columns : list of str

    unused_columns : list of str

    bool_columns : list of str
        Columns which have boolean values (can be null).

    decorated_columns : list of str
        Numerical columns with units (in, hp, lb-ft, etc.) attached.

    discrete_columns : list of str
        Numerical discrete columns.

    categorical_columns : list of str
        Columns with categorical values which are not binary.

    binary_columns : list of str
        Columns with binary categorical values.

    numeric_columns : list of str

    numeric_full_columns : list of str
        Numeric columns with 0% null values.

    fuel_columns : list of str
        Columns related to the fuel consumption of a car.

    engine_columns : list of str
        Columns related to the engine of a car. The default list is
        determined from correlation.

    usage_columns : list of str
        Columns related to the usage of a car.

    dimension_columns : list of str
        Columns related to the dimensions of a car.

    exclude_columns : list of str
        Columns to be excluded in the final features matrix.

    default_final_columns : list of str
        Columns in the final feature matrix using all default options.

    """
    columns = ['vin', 'back_legroom', 'bed', 'bed_height', 'bed_length',
               'body_type', 'cabin', 'city', 'city_fuel_economy',
               'combine_fuel_economy', 'daysonmarket', 'dealer_zip',
               'description', 'engine_cylinders', 'engine_displacement',
               'engine_type', 'exterior_color', 'fleet', 'frame_damaged',
               'franchise_dealer', 'franchise_make', 'front_legroom',
               'fuel_tank_volume', 'fuel_type', 'has_accidents', 'height',
               'highway_fuel_economy', 'horsepower', 'interior_color', 'isCab',
               'is_certified', 'is_cpo', 'is_new', 'is_oemcpo', 'latitude',
               'length', 'listed_date', 'listing_color', 'listing_id',
               'longitude', 'main_picture_url', 'major_options', 'make_name',
               'maximum_seating', 'mileage', 'model_name', 'owner_count',
               'power', 'price', 'salvage', 'savings_amount', 'seller_rating',
               'sp_id', 'sp_name', 'theft_title', 'torque', 'transmission',
               'transmission_display', 'trimId', 'trim_name',
               'vehicle_damage_category', 'wheel_system',
               'wheel_system_display', 'wheelbase', 'width', 'year']

    columns_dtype = {'vin': 'object', 'back_legroom': 'object',
                     'bed': 'object', 'bed_height': 'object',
                     'bed_length': 'object', 'body_type': 'object',
                     'cabin': 'object', 'city': 'object',
                     'city_fuel_economy': 'float64',
                     'combine_fuel_economy': 'float64',
                     'daysonmarket': 'int64', 'dealer_zip': 'object',
                     'description': 'object', 'engine_cylinders': 'object',
                     'engine_displacement': 'float64', 'engine_type': 'object',
                     'exterior_color': 'object', 'fleet': 'object',
                     'frame_damaged': 'object', 'franchise_dealer': 'bool',
                     'franchise_make': 'object', 'front_legroom': 'object',
                     'fuel_tank_volume': 'object', 'fuel_type': 'object',
                     'has_accidents': 'object', 'height': 'object',
                     'highway_fuel_economy': 'float64',
                     'horsepower': 'float64', 'interior_color': 'object',
                     'isCab': 'object', 'is_certified': 'float64',
                     'is_cpo': 'object', 'is_new': 'bool',
                     'is_oemcpo': 'object', 'latitude': 'float64',
                     'length': 'object', 'listed_date': 'object',
                     'listing_color': 'object', 'listing_id': 'int64',
                     'longitude': 'float64', 'main_picture_url': 'object',
                     'major_options': 'object', 'make_name': 'object',
                     'maximum_seating': 'object', 'mileage': 'float64',
                     'model_name': 'object', 'owner_count': 'float64',
                     'power': 'object', 'price': 'float64',
                     'salvage': 'object', 'savings_amount': 'int64',
                     'seller_rating': 'float64', 'sp_id': 'float64',
                     'sp_name': 'object', 'theft_title': 'object',
                     'torque': 'object', 'transmission': 'object',
                     'transmission_display': 'object', 'trimId': 'object',
                     'trim_name': 'object',
                     'vehicle_damage_category': 'float64',
                     'wheel_system': 'object',
                     'wheel_system_display': 'object', 'wheelbase': 'object',
                     'width': 'object', 'year': 'int64'}

    null_columns = ["combine_fuel_economy", "is_certified",
                    "vehicle_damage_category", "bed_height", "bed_length"]
    useless_columns = ["description", "main_picture_url"]
    unused_columns = ["city", "model_name", "interior_color", "major_options",
                      "trimId", 'trim_name', 'sp_name', 'dealer_zip', "sp_id",
                      'latitude', 'longitude', 'listing_id', "vin",
                      "exterior_color"]

    bool_columns = ["fleet", "frame_damaged", "has_accidents", "isCab",
                    "is_cpo", "is_oemcpo", "salvage", "theft_title"]
    decorated_columns = ["back_legroom", "front_legroom", "fuel_tank_volume",
                         "height", "length", "maximum_seating", "wheelbase",
                         "width", "power", "torque"]
    discrete_columns = ["daysonmarket", "maximum_seating", "owner_count",
                        "savings_amount", "year"]

    categorical_columns = ["bed", "body_type", "cabin", "engine_cylinders",
                           "engine_type", "franchise_make", "fuel_type",
                           "make_name", "transmission", "transmission_display",
                           "wheel_system", "wheel_system_display",
                           "listing_color"]
    binary_columns = ["isTruck", "is_new", "franchise_dealer"]

    numeric_columns = ["back_legroom", "city_fuel_economy", "daysonmarket",
                       "engine_displacement", "front_legroom",
                       "fuel_tank_volume", "height", "highway_fuel_economy",
                       "horsepower", "length", "maximum_seating", "mileage",
                       "owner_count", "power", "powerRPM", "savings_amount",
                       "seller_rating", "torque", "torqueRPM", "wheelbase",
                       "width", "year", "car_age"]

    # full: no null values
    numeric_full_columns = ["daysonmarket", "price", "savings_amount", "year",
                            "car_age"]

    fuel_columns = ["city_fuel_economy", "highway_fuel_economy"]
    engine_columns = ["engine_displacement", "fuel_tank_volume", "horsepower",
                      "power", "torque", "wheelbase", "length"]
    usage_columns = ["car_age", "mileage"]
    dimension_columns = ['height', 'width', 'back_legroom', 'front_legroom']

    exclude_columns = ["bed", "body_type", "cabin", "engine_cylinders",
                       "engine_type", "franchise_make", "fuel_type",
                       "make_name", "transmission", "transmission_display",
                       "wheel_system", "wheel_system_display", "year",
                       "listing_color"]


class FeatureDropper(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Drops features from dataframe.

    The input is a list of feature/column names to be dropped from the
    dataframe. By default, this transformer drops `null_columns`,
    `useless_columns` and `unused_columns` from the `Info` class.

    Parameters
    ----------
    remove_cols : list of str, default=None
        Names of features/columns to drop.

    Attributes
    ----------
    remove_cols_ : list of str
        Names of features/columns to drop. When `remove_cols` is not
        specified, the list contains a combination of `null_columns`,
        `useless_columns` and `unused_columns` from the `Info`
        class.

    """

    def __init__(self, remove_cols=None):
        self.remove_cols = remove_cols

    def fit(self, X, y=None):
        if self.remove_cols is None:
            self.remove_cols_ = (self.null_columns + self.useless_columns
                                 + self.unused_columns)
        else:
            self.remove_cols_ = self.remove_cols
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.remove_cols_, errors="ignore")


class TruckCreator(base.BaseEstimator, base.TransformerMixin):
    """
    Adds `isTruck` column based on the `bed` column.

    The value of `isTruck` is True unless `bed` is null.

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.insert(2, "isTruck", X["bed"].notna())
        return X


class CabinModifier(base.BaseEstimator, base.TransformerMixin):
    """
    Changes data in the `cabin` column to `"none"` when `isTruck` is
    False.

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[(X["isTruck"] == False) & (X["cabin"].isna()), "cabin"] = "none"
        return X


class BoolConverter(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Converts columns from object to boolean dtype.

    Parameters
    ----------
    boolean_cols : list of str, default=None
        Columns to convert from object dtype to boolean dtype.

    Attributes
    ----------
    boolean_cols_ : list of str
        Columns to convert from object dtype to boolean dtype. When
        `boolean_cols` is not specified, `bool_columns` from the
        `Info` class is used.

    """

    def __init__(self, boolean_cols=None):
        self.boolean_cols = boolean_cols

    def fit(self, X, y=None):
        if self.boolean_cols is None:
            self.boolean_cols_ = self.bool_columns
        else:
            self.boolean_cols_ = self.boolean_cols
        return self

    def transform(self, X, y=None):
        X[self.boolean_cols_] = X[self.boolean_cols_].replace(
            {"True": True, "False": False})
        X[self.boolean_cols_] = X[self.boolean_cols_].astype("boolean")
        return X


class NumericTransformer(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Strips units from column contents and splits the RPM of power and
    torque into separate columns.

    Parameters
    ----------
    strip_cols : list of str, default=None
        Numeric columns with units (in, lb-ft, etc.) to strip.

    Attributes
    ----------
    strip_cols_ : list of str
        Numeric columns with units (in, lb-ft, etc.) to strip. When
        `strip_cols` is not specified, `decorated_columns` from the
        `Info` class is used.

    """

    def __init__(self, strip_cols=None):
        self.strip_cols = strip_cols

    def fit(self, X, y=None):
        if self.strip_cols is None:
            self.strip_cols_ = self.decorated_columns
        else:
            self.strip_cols_ = self.strip_cols
        return self

    def transform(self, X, y=None):
        # Strip units from certain columns
        for col in [col for col in self.strip_cols_
                    if col not in ["power", "torque"]]:
            X[col] = self.strip_units(data=X, col=col).copy()

        # Separate "power" and "torque" columns into separate value and
        # RPM columns
        for col, unit in [("power", "hp"), ("torque", "lb-ft")]:
            rpm_col = f"{col}RPM"

            value_series = X[col].apply(
                lambda s: s[: s.find(unit)].strip().replace(",", "")
                if pd.notna(s) and "--" not in s else None)
            rpm_series = X[col].apply(self.extract_rpm)

            X[col] = pd.to_numeric(value_series)
            X[rpm_col] = pd.to_numeric(rpm_series)

            X.insert(X.columns.get_loc(col) + 1, rpm_col, X.pop(rpm_col))

        return X

    def strip_units(self, data, col):
        series = data[col]
        # Some columns contains "--"
        series = series.apply(
            lambda s: re.sub(r"[^0-9.]", "", s)
            if pd.notna(s) and "--" not in s else None)
        series = pd.to_numeric(series)
        return series

    def extract_rpm(self, s):
        if pd.notna(s) and "--" not in s and "@" in s:
            return s[s.find("@")+1: s.find("RPM")].strip().replace(",", "")
        else:
            return None


class IntegerConverter(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Converts columns to Int64 dtype (accepts null values).

    Parameters
    ----------
    int_cols : list of str, default=None
        Discrete numeric columns to convert.

    Attributes
    ----------
    int_cols_ : list of str
        Discrete numeric columns to convert. When `int_cols` is not
        specified, `discrete_columns` from the `Info` class is used.

    """

    def __init__(self, int_cols=None):
        self.int_cols = int_cols

    def fit(self, X, y=None):
        if self.int_cols is None:
            self.int_cols_ = self.discrete_columns
        else:
            self.int_cols_ = self.int_cols
        return self

    def transform(self, X, y=None):
        X[self.int_cols_] = np.rint(X[self.int_cols_]).astype("Int64")
        return X


class CategoryConverter(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Converts columns to category dtype.

    Parameters
    ----------
    cat_cols : list of str, default=None
        Categorical columns to convert.

    bool_cat_cols : list of str, default=None
        Binary categorical columns to convert.

    all_cat_cols : list of str, default=None
        Both binary and nonbinary categorical columns combined.

    Attributes
    ----------
    cat_cols_ : list of str
        Categorical columns to convert. When `cat_cols` is not
        specified, `categorical_columns` from the `Info` class is
        used.

    bool_cat_cols_ : list of str
        Binary categorical columns to convert. When `bool_cat_cols` is
        not specified, `binary_columns` from the `Info` class is
        used.

    all_cat_cols_ : list of str
        Both binary and nonbinary categorical columns combined.

    """

    def __init__(self, cat_cols=None, bool_cat_cols=None, all_cat_cols=None):
        self.cat_cols = cat_cols
        self.bool_cat_cols = bool_cat_cols
        self.all_cat_cols = all_cat_cols

    def fit(self, X, y=None):
        if self.cat_cols is None:
            self.cat_cols_ = self.categorical_columns
        else:
            self.cat_cols_ = self.cat_cols

        if self.bool_cat_cols is None:
            self.bool_cat_cols_ = self.binary_columns
        else:
            self.bool_cat_cols_ = self.bool_cat_cols

        if self.all_cat_cols is None:
            self.all_cat_cols_ = self.cat_cols_ + self.bool_cat_cols_
        else:
            self.all_cat_cols_ = self.all_cat_cols
        return self

    def transform(self, X, y=None):
        X[self.all_cat_cols_] = X[self.all_cat_cols_].astype("category")
        return X


class DateConverter(base.BaseEstimator, base.TransformerMixin):
    """
    Converts the `listed_date` column to datetime64[ns] dtype.

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["listed_date"] = X["listed_date"].astype("datetime64[ns]")
        return X


class CarAgeCreator(base.BaseEstimator, base.TransformerMixin):
    """
    Adds `car_age` column based on the `year` and `listed_date` columns.

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["car_age"] = pd.to_numeric(
            X["listed_date"].dt.year).sub(X["year"], axis=0)
        return X


class NumericImputer(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Imputes null values for numeric columns.

    Parameters
    ----------
    imputer : sklearn.impute, default=sklearn.impute.SimpleImputer()
        Used to impute missing numeric values.

    numeric_cols : list of str, default=None
        All numeric columns.

    numeric_full_cols : list of str, default=None
        Numeric columns without any missing values.

    Attributes
    ----------
    numeric_cols_ : list of str
        All numeric columns. When `numeric_cols` is not specified,
        `numeric_columns` from the `Info` class is used.

    numeric_full_cols_ : list of str
        Numeric columns without any missing values. When
        `numeric_full_cols` is not specified, `numeric_full_columns`
        from the `Info` class is used.

    numeric_null_cols_ : list of str
        Numeric columns with missing values.

    """

    def __init__(self, imputer=SimpleImputer(), numeric_cols=None,
                 numeric_full_cols=None):
        self.imputer = imputer
        self.numeric_cols = numeric_cols
        self.numeric_full_cols = numeric_full_cols

    def fit(self, X, y=None):
        if self.numeric_cols is None:
            self.numeric_cols_ = self.numeric_columns
        else:
            self.numeric_cols_ = self.numeric_cols

        if self.numeric_full_cols is None:
            self.numeric_full_cols_ = self.numeric_full_columns
        else:
            self.numeric_full_cols_ = self.numeric_full_cols

        self.numeric_null_cols_ = [col for col in self.numeric_cols_
                                   if col not in self.numeric_full_cols_]

        self.imputer.fit(X[self.numeric_null_cols_])
        return self

    def transform(self, X, y=None):
        X[self.numeric_null_cols_] = self.imputer.transform(
            X[self.numeric_null_cols_])
        return X


class BoolImputer(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Imputes null values for boolean columns.

    Parameters
    ----------
    imputer : sklearn.impute,
    default=sklearn.impute.SimpleImputer(strategy="most_frquent")
        Used to impute missing values.

    bool_cols : list of str
        Columns with boolean values.

    Attributes
    ----------
    bool_cols_ : list of str
        Columns with boolean values. When `bool_cols` is not specified,
        `bool_columns` from the `Info` class is used.

    """

    def __init__(self, imputer=SimpleImputer(strategy="most_frequent"),
                 bool_cols=None):
        self.imputer = imputer
        self.bool_cols = bool_cols

    def fit(self, X, y=None):
        if self.bool_cols is None:
            self.bool_cols_ = self.bool_columns
        else:
            self.bool_cols_ = self.bool_cols

        self.imputer.fit(X[self.bool_cols_])
        return self

    def transform(self, X, y=None):
        X[['is_cpo', 'is_oemcpo']] = X[['is_cpo', 'is_oemcpo']].fillna(False)
        X[self.bool_cols_] = self.imputer.transform(X[self.bool_cols_])
        return X


class FuelPCA(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Reduces the dimension of fuel-related columns using principal
    component analysis.

    Parameters
    ----------
    n_components : int, default=1
        Passed to `PCA()`.

    fuel_cols : list of str, default=None
        Features to do principal component analysis from.

    random_state : int, default=None
        Passed to `PCA()`.

    Attributes
    ----------
    fuel_cols_ : list of str
        Features to do principal component analysis from. When
        `fuel_cols` is unspecified, `fuel_columns` from the `Info` class
        is used.

    pca : skelarn.decomposition.PCA

    """

    def __init__(self, n_components=1, fuel_cols=None, random_state=None):
        self.n_components = n_components
        self.fuel_cols = fuel_cols
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.fuel_cols is None:
            self.fuel_cols_ = self.fuel_columns
        else:
            self.fuel_cols_ = self.fuel_cols

        self.pca = PCA(self.n_components, random_state=self.random_state)
        self.pca.fit(X[self.fuel_cols_])
        return self

    def transform(self, X, y=None):
        temp_df = pd.DataFrame(
            data=self.pca.transform(X[self.fuel_cols_]),
            index=X.index,
            columns=[f"pca_fuel_economy_{i+1}"
                     for i in range(self.n_components)])
        return pd.concat([X, temp_df], axis=1)


class EngineScaler(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Scales the engine-related columns.

    Parameters
    ----------
    scaler : sklearn.preprocessing.MinMaxScaler() or
    sklearn.preprocessing.StandardScaler(),
    default=sklearn.preprocessing.MinMaxScaler()

    engine_cols : list of str, default=None
        Features to scale.

    Attributes
    ----------
    engine_cols_ : list of str
        Features to scale. When `engine_cols` is not specified,
        `engine_columns` from the `Info` class is used.

    """

    def __init__(self, scaler=MinMaxScaler(), engine_cols=None):
        self.scaler = scaler
        self.engine_cols = engine_cols

    def fit(self, X, y=None):
        if self.engine_cols is None:
            self.engine_cols_ = self.engine_columns
        else:
            self.engine_cols_ = self.engine_cols

        self.scaler.fit(X[self.engine_cols_])
        return self

    def transform(self, X, y=None):
        engine_normal_df = pd.DataFrame(
            data=self.scaler.transform(X[self.engine_cols_]),
            columns=self.engine_cols_)
        engine_normal_df = engine_normal_df.set_index(X.index)
        X[self.engine_cols_] = engine_normal_df.copy()
        return X


class EnginePCA(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Reduces the dimension of engine-related columns using principal
    component analysis.

    Parameters
    ----------
    n_components : int, default=1
        Passed to `PCA()`.

    engine_cols : list of str, default=None
        Features to do principal component analysis from.

    random_state : int, default=None
        Passed to `PCA()`.

    Attributes
    ----------
    engine_cols_ : list of str
        Features to do principal component analysis from. When
        `engine_cols` is unspecified, `engine_columns` from the `Info`
        class is used.

    pca : skelarn.decomposition.PCA

    """

    def __init__(self, n_components=1, engine_cols=None, random_state=None):
        self.n_components = n_components
        self.engine_cols = engine_cols
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.engine_cols is None:
            self.engine_cols_ = self.engine_columns
        else:
            self.engine_cols_ = self.engine_cols

        self.pca = PCA(self.n_components, random_state=self.random_state)
        self.pca.fit(X[self.engine_cols_])
        return self

    def transform(self, X, y=None):
        temp_df = pd.DataFrame(
            data=self.pca.transform(X[self.engine_cols_]),
            index=X.index,
            columns=[f"pca_engine_{i+1}" for i in range(self.n_components)])
        return pd.concat([X, temp_df], axis=1)


class CarUsagePCA(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Reduces the dimension of car usage-related columns using principal
    component analysis.

    Parameters
    ----------
    n_components : int, default=1
        Passed to `PCA()`.

    usage_cols : list of str, default=None
        Features to do principal component analysis from.

    random_state : int, default=None
        Passed to `PCA()`.

    Attributes
    ----------
    usage_cols_ : list of str
        Features to do principal component analysis from. When
        `usage_cols` is unspecified, `usage_columns` from the `Info`
        class is used.

    pca : skelarn.decomposition.PCA

    """

    def __init__(self, n_components=1, usage_cols=None, random_state=None):
        self.n_components = n_components
        self.usage_cols = usage_cols
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.usage_cols is None:
            self.usage_cols_ = self.usage_columns
        else:
            self.usage_cols_ = self.usage_cols

        self.pca = PCA(self.n_components, random_state=self.random_state)
        self.pca.fit(X[self.usage_cols_])
        return self

    def transform(self, X, y=None):
        temp_df = pd.DataFrame(
            data=self.pca.transform(X[self.usage_cols_]),
            index=X.index,
            columns=[f"pca_car_usage_{i+1}"
                     for i in range(self.n_components)])
        return pd.concat([X, temp_df], axis=1)


class CarSpaceAvg(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Reduces the car dimensions' columns into one column by taking the
    weighted average of the price.

    Parameters
    ----------
    space_cols : list of str, default=None
        Features to do weighted average on.

    Attributes
    ----------
    space_cols_ : list of str
        Features to do weighted average on. When `space_cols` is
        unspecified, `dimension_columns` from the `Info` class is used.

    weight: dict
        Features in `space_cols_` and their weights.

    """

    def __init__(self, space_cols=None):
        self.space_cols = space_cols

    def fit(self, X, y=None):
        """
        Fits the weights based on X and y.

        Raises
        ------
        Exception
            If `y` is not specified.

        """

        try:
            if y is None:
                raise Exception("y must not be None.")
        except Exception:
            raise

        self.price_df = pd.DataFrame(y, columns=["price"])
        self.combined_df = pd.concat([X, y], axis=1)

        self.weight = dict()
        if self.space_cols is None:
            self.space_cols_ = self.dimension_columns
        else:
            self.space_cols_ = self.space_cols

        self.temp_df = self.combined_df[self.space_cols_]
        self.temp_df = ((self.temp_df - self.temp_df.min())
                        / (self.temp_df.max() - self.temp_df.min()))

        self.corr_df = self.combined_df[self.space_cols_ + ["price"]].corr()
        self.weight = self.corr_df.iloc[:-1, -1].to_dict()
        return self

    def transform(self, X, y=None):
        X["avg_car_space"] = (
            X[self.space_cols_] * pd.Series(self.weight)).sum(axis=1)
        return X


class CategoryEncoder(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Encodes categorical columns to numeric using one hot encoding and
    principal component analysis.

    Parameters
    ----------
    n_components : int, default=1
        Passed to `PCA()`.

    cat_cols : list of str, default=None
        Categorical features to encode.

    random_state : int, default=None
        Passed to `PCA()`.

    Attributes
    ----------
    cat_cols_ : list of str
        Categorical features to encode. If `cat_cols` is unspecified,
        `categorical_columns` from the `Info` class is used.

    encoders : dict of str and sklearn.preprocessing.OneHotEncoder
        Key-value pairs of column names in `cat_cols_` and their
        respective encoders.

    pcas : dict of str and skelarn.decomposition.PCA
        Key-value pairs of column names in `cat_cols_` and their
        respective PCA objects.

    """

    def __init__(self, n_components=1, cat_cols=None, random_state=None):
        self.n_components = n_components
        self.cat_cols = cat_cols
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.cat_cols is None:
            self.cat_cols_ = self.categorical_columns
        else:
            self.cat_cols_ = self.cat_cols

        X[self.cat_cols_] = X[self.cat_cols_].astype("object")
        X[self.cat_cols_] = X[self.cat_cols_].fillna("none")
        X[self.cat_cols_] = X[self.cat_cols_].astype("category")

        self.encoders = dict()
        self.pcas = dict()
        for col in self.cat_cols_:
            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(X[[col]])
            self.encoders[col] = encoder

            col_encoded = encoder.transform(X[[col]])
            col_encoded_df = pd.DataFrame(
                data=col_encoded.toarray(),
                columns=encoder.get_feature_names_out([col]))

            pca = PCA(self.n_components, random_state=self.random_state)
            pca.fit(col_encoded_df)
            self.pcas[col] = pca

        return self

    def transform(self, X, y=None):
        X[self.cat_cols_] = X[self.cat_cols_].astype("object")
        X[self.cat_cols_] = X[self.cat_cols_].fillna("none")
        X[self.cat_cols_] = X[self.cat_cols_].astype("category")

        for col in self.cat_cols_:
            encoder = self.encoders[col]
            col_encoded = encoder.transform(X[[col]])
            col_encoded_df = pd.DataFrame(
                data=col_encoded.toarray(),
                columns=encoder.get_feature_names_out([col]))

            pca = self.pcas[col]
            col_encoded_df = pd.DataFrame(
                data=pca.transform(col_encoded_df),
                index=X.index,
                columns=[f"{col}_encoded_{i+1}"
                         for i in range(self.n_components)])
            X = pd.concat([X, col_encoded_df], axis=1)

        return X


class FinalFeatureSelector(base.BaseEstimator, base.TransformerMixin, Info):
    """
    Selects the featues for the final feature matrix.

    Parameters
    ----------
    exclude_cols : "auto", None, or list of str, default="auto"
        Columns to exclude in the final feature matrix.

    final_cols : list of str, default=None
        Columns to include in the final feature matrix.

    Attributes
    ----------
    exclude_cols_ : list of str
        Columns to exclude in the final feature matrix. If `final_cols`
        is specified, `exclude_cols_` will be defined as the complement
        of `final_cols`.

    """

    def __init__(self, exclude_cols="auto", final_cols=None):
        self.exclude_cols = exclude_cols
        self.final_cols = final_cols

    def fit(self, X, y=None):
        if self.final_cols is not None:
            self.exclude_cols_ = [col for col in X.columns
                                  if col not in self.final_cols]
        elif self.exclude_cols == "auto":
            self.exclude_cols_ = (self.exclude_columns + self.fuel_columns
                                  + self.engine_columns + self.usage_columns
                                  + self.dimension_columns)
        elif self.exclude_cols is None:
            self.exclude_cols_ = self.exclude_columns
        else:
            self.exclude_cols_ = self.exclude_cols

        return self

    def transform(self, X, y=None):
        X = X.drop(columns=self.exclude_cols_)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col])
        return X


def convert_to_feather(csv_filepath, names=Info.columns,
                       dtype=Info.columns_dtype, random_state=None, **kwargs):
    csv_df = pd.read_csv(csv_filepath, names=names, dtype=dtype, header=0,
                         sep=",", engine="c")
    for k, v in kwargs.items():
        try:
            df = csv_df.sample(v, random_state=random_state)
        except TypeError:
            print("Value in **kwargs must be an integer")
            raise
        except ValueError:
            print("Value must not be larger than the number of rows")
            raise
        else:
            feather.write_feather(df, f"{csv_filepath[:-4]}_{k}.feather")


def remove_null_rows(data, columns):
    """
    Removes all rows from `data` with null values for columns listed in
    `columns`.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe to remove rows from.

    columns : list of str
        Column names from which to remove rows with null values from.

    Returns
    -------
    pandas.DataFrame
        Dataframe with removed rows.

    """

    return data[data[columns].notna().all(1)]


def make_used_cars_transformer(
    imputer=SimpleImputer(strategy="mean"),
    n_components=[1, 1, 1],
    random_state=None
):
    """Make a pipeline without a model."""

    return Pipeline([
        ('remove_null', FeatureDropper()),
        ('create_isTruck_var', TruckCreator()),
        ('modify_cabin_var', CabinModifier()),
        ('convert_bool', BoolConverter()),
        ('convert_numeric', NumericTransformer()),
        ('convert_integer', IntegerConverter()),
        ('convert_category', CategoryConverter()),
        ('convert_datetime', DateConverter()),
        ('creat_car_age_var', CarAgeCreator()),
        ('impute_numeric', NumericImputer(imputer=imputer)),
        ('impute_bool', BoolImputer()),
        ('reduce_fuel_dim', FuelPCA(n_components=n_components[0],
                                    random_state=random_state)),
        ('scale_engine_var', EngineScaler()),
        ('reduce_engine_dim', EnginePCA(n_components=n_components[1],
                                        random_state=random_state)),
        ('reduce_usage_dim', CarUsagePCA(n_components=n_components[2],
                                         random_state=random_state)),
        ('reduce_space_dim', CarSpaceAvg()),
        ('encode_category', CategoryEncoder(random_state=random_state)),
    ])


def make_used_cars_pipeline(
    model=LinearRegression(),
    scaler=None,
    imputer=SimpleImputer(strategy="mean"),
    n_components=[1, 1, 1],
    random_state=None,
    **kwargs
):
    """Make a pipeline with a model included."""

    internal = make_used_cars_transformer(
        imputer=imputer, n_components=n_components, random_state=random_state)
    if scaler is None:
        return Pipeline([
            ('internal', internal),
            ('select_features', FinalFeatureSelector(**kwargs)),
            ('model', model),
        ])
    else:
        return Pipeline([
            ('internal', internal),
            ('select_features', FinalFeatureSelector(**kwargs)),
            ('scaler', scaler),
            ('model', model),
        ])


def write_test_results(filepath, test_results, models, pickle=False):
    """Save `test_results` dictionary and `models` list to a joblib file
    and save formatted version to a text file."""

    sep = filepath.rfind("/")
    dirpath = filepath[: sep+1]
    filename = filepath[sep+1:]

    if pickle:
        with open(f"{dirpath}pickle_{filename}.joblib", "wb") as f:
            joblib.dump((models, test_results), f)
    with open(f"{filepath}.txt", "w") as f:
        for label, value in test_results.items():
            if label == "info":
                for k, v in value.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
            elif label == "duration":
                f.write(f"{label}: {value:.3f} s\n")
            else:
                f.write(f"{label}:\n")
                for k, v in value.items():
                    if k == "model":
                        f.write(f"\t{k}: {v}\n")
                    elif k == "scores":
                        f.write(f"\t{k}:\n")
                        for score_label, score_arr in v.items():
                            f.write(
                                f"\t\tAverage {score_label}: {score_arr.mean():.3f}")
                            if score_label in ["fit_time", "score_time"]:
                                f.write(" s\n")
                            else:
                                f.write("\n")
                    elif k == "duration":
                        f.write(f"\tTotal {k}: {v:.3f} s\n")
                f.write("\n")


def run_test(
    X, y, n_splits, sample_size, test_size, random_state,
    models=[("Linear Regression", LinearRegression())],
    imputers=[SimpleImputer(strategy="mean")],
    cv=None, filepath=None, **kwargs
):
    if cv is None:
        cv = ShuffleSplit(n_splits, test_size=test_size,
                          random_state=random_state)

    test_results = {
        "info": {
            "sample_size": sample_size,
            "test_size": test_size,
            "n_splits": n_splits,
            "random_state": random_state,
        },
    }

    if "final_cols" in kwargs:
        test_results["info"]["features_selected"] = kwargs["final_cols"]
    elif "exclude_cols" in kwargs:
        test_results["info"]["features_excluded"] = kwargs["exclude_cols"]
    else:
        test_results["info"]["features_excluded"] = (
            Info.exclude_columns + Info.fuel_columns + Info.engine_columns
            + Info.usage_columns + Info.dimension_columns
        )

    global_start = time.time()

    for label, model in models:
        for imputer in imputers:
            test_results[f"{label}_{imputer}"] = dict()
            test_results[f"{label}_{imputer}"]["model"] = model
            start_time = time.time()
            test_results[f"{label}_{imputer}"]["scores"] = cross_validate(
                X=X, y=y, cv=cv, verbose=0, return_train_score=True, n_jobs=-1,
                estimator=make_used_cars_pipeline(
                    model=model,
                    imputer=imputer,
                    scaler=(None if label != "Ridge Regression"
                            else StandardScaler()),
                    random_state=random_state,
                    **kwargs),
                scoring=["r2", "neg_mean_absolute_error",
                         "neg_root_mean_squared_error",
                         "neg_mean_absolute_percentage_error"])
            end_time = time.time()
            test_results[f"{label}_{imputer}"]["duration"] = (end_time
                                                              - start_time)

    global_end = time.time()
    test_results["duration"] = global_end - global_start

    if filepath is not None:
        write_test_results(filepath, test_results, models)

    return test_results
