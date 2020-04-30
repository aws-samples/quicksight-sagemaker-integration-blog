from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd
import logging

from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    'State',
    'Account Length',
    'Area Code',
    'Phone',
    "Int'l Plan",
    'VMail Plan',
    'VMail Message',
    'Day Mins',
    'Day Calls',
    'Day Charge',
    'Eve Mins',
    'Eve Calls',
    'Eve Charge',
    'Night Mins',
    'Night Calls',
    'Night Charge',
    'Intl Mins',
    'Intl Calls',
    'Intl Charge',
    'CustServ Calls'] 

label_column = 'Churn?'

feature_columns_dtype = {
    'State' :  str,
    'Account Length' :  np.int64,
    'Area Code' :  str,
    'Phone' :  str,
    "Int'l Plan" :  str,
    'VMail Plan' :  str,
    'VMail Message' :  np.int64,
    'Day Mins' :  np.float64,
    'Day Calls' :  np.int64,
    'Day Charge' :  np.float64,
    'Eve Mins' :  np.float64,
    'Eve Calls' :  np.int64,
    'Eve Charge' :  np.float64,
    'Night Mins' :  np.float64,
    'Night Calls' :  np.int64,
    'Night Charge' :  np.float64,
    'Intl Mins' :  np.float64,
    'Intl Calls' :  np.int64,
    'Intl Charge' :  np.float64,
    'CustServ Calls' :  np.int64}

label_column_dtype = {'Churn?': str}  

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def _is_inverse_label_transform():
    """Returns True if if it's running in inverse label transform."""
    return os.getenv('TRANSFORM_MODE') == 'inverse-label-transform'

def _is_feature_transform():
    """Returns True if it's running in feature transform mode."""
    return os.getenv('TRANSFORM_MODE') == 'feature-transform'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])


    args = parser.parse_args()

    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [ pd.read_csv(
        file, 
        header=None, 
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)) for file in input_files ]
    concat_data = pd.concat(raw_data)

    numeric_features = list([
    'Account Length',
    'VMail Message',
    'Day Mins',
    'Day Calls',
    'Eve Mins',
    'Eve Calls',
    'Night Mins',
    'Night Calls',
    'Intl Mins',
    'Intl Calls',
    'CustServ Calls'])


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['State','Area Code',"Int'l Plan",'VMail Plan']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder="drop")

    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    
def input_fn(input_data, request_content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    
    
    content_type = request_content_type.lower(
    ) if request_content_type else "text/csv"
    content_type = content_type.split(";")[0].strip()
    
    
    if isinstance(input_data, str):
        str_buffer = input_data
    else:
        str_buffer = str(input_data,'utf-8')
    

    if _is_feature_transform():
        if content_type == 'text/csv':
            # Read the raw input data as CSV.
            df = pd.read_csv(StringIO(input_data),  header=None)
            if len(df.columns) == len(feature_columns_names) + 1:
                # This is a labelled example, includes the  label
                df.columns = feature_columns_names + [label_column]
            elif len(df.columns) == len(feature_columns_names):
                # This is an unlabelled example.
                df.columns = feature_columns_names
            return df
        else:
            raise ValueError("{} not supported by script!".format(content_type))
    
    
    if _is_inverse_label_transform():
        if (content_type == 'text/csv' or content_type == 'text/csv; charset=utf-8'):
            # Read the raw input data as CSV.
            df = pd.read_csv(StringIO(str_buffer),  header=None)
            logging.info(f"Shape of the requested data: '{df.shape}'")
            return df
        else:
            raise ValueError("{} not supported by script!".format(content_type))
            
            
def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    
    accept = 'text/csv'
    if type(prediction) is not np.ndarray:
        prediction=prediction.toarray()
    
   
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """

    
    if _is_feature_transform():
        features = model.transform(input_data)


        if label_column in input_data:
            # Return the label (as the first column) and the set of features.
            return np.insert(features.toarray(), 0, pd.get_dummies(input_data[label_column])['True.'], axis=1)
        else:
            # Return only the set of features
            return features
    
    if _is_inverse_label_transform():
        features = input_data.iloc[:,0]>0.5
        features = features.values
        return features
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    if _is_feature_transform():
        preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
        return preprocessor