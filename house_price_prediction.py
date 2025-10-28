import os
import joblib
import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE= "pipeline.pkl"

def build_pipeline(num_attributes,cat_attributes):
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy = "median")),
                         ("scaler", StandardScaler())])


#cat pipeline

    cat_pipeline = Pipeline([("one_hot",OneHotEncoder(handle_unknown= "ignore"))])


    full_pipeline = ColumnTransformer(
        [
            ("num_att",num_pipeline,num_attributes),
            ("cat",cat_pipeline,cat_attributes)
        ]
    )
    return full_pipeline
if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")
    housing['income_cat']= pd.cut(housing['median_income'],bins =[0.0,1.5,3.0,4.5,6.0, np.inf],labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index,test_index in split.split(housing,housing["income_cat"]):
        Strat_train_set = housing.loc[train_index].drop('income_cat',axis = 1)
        Strat_test_set = housing.loc[test_index].drop('income_cat',axis = 1)
    housing= Strat_train_set.copy()

    #Seprate labels and features

    housing_label = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value',axis = 1)

    # seprate numerical and catgerocial columns

    num_attributes = housing_features.drop("ocean_proximity",axis= 1).columns.tolist()
    cat_attributes =["ocean_proximity"]
    
    pipeline  = build_pipeline(num_attributes,cat_attributes)
    housing_prepared= pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(n_estimators=50,random_state=42)
    model.fit(housing_prepared,housing_label)
    
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
else:
    print("Model already trained. Ready for Streamlit app!")