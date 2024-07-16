import os
import pandas as pd
import pickle

from ml.data import process_data
from ml.model import compute_model_metrics, inference

def slice_metrics(model, encoder, lb, data, slice_feature, categorical_features=[]):
    """
    Computes performance on model slices.
    Inputs:
        model : Trained model
        encoder : sklearn.preprocessing._encoders.OneHotEncoder
        lb : Trained sklearn LabelBinarizer
        data : Dataframe containing the features and label.
        slice_feature: feature used to make slices.
        categorical_features: list[str] List containing the names of the categorical features.
    """

    X, y, encoder, lb = process_data(data, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb)
    predictions = inference(model, X)

    with open('slice_output.txt', 'w') as f:
        for slice_value in data[slice_feature].unique():
            slice_index = data.index[data[slice_feature] == slice_value]
            
            f.write(str(slice_feature)+' = '+str(slice_value)+'\n')
            f.write('data size:{}\n'.format(len(slice_index)))
            f.write('precision: {}, recall: {}, fbeta: {}\n'.format(
                *compute_model_metrics(y[slice_index], predictions[slice_index])
            )
            )
            f.write('============================================================================================'+'\n')


if __name__ == '__main__':
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    file_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(file_dir, '../data/clean_census.csv'))

    model_path = os.path.join(file_dir, '../model/model.pkl')
    model = pickle.load(open(model_path, 'rb'))

    encoder_path = os.path.join(file_dir, '../model/encoder.pkl')
    encoder = pickle.load(open(encoder_path, 'rb'))

    lb_path = os.path.join(file_dir, '../model/label_binarizer.pkl')
    lb = pickle.load(open(lb_path, 'rb'))

    slice_metrics(model, encoder, lb, data, 'workclass', categorical_features=cat_features)