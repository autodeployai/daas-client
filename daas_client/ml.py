#
# Copyright (c) 2017-2019 AutoDeploy AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import json
import os
import pandas as pd

FUNCTION_NAME_CLASSIFICATION = 'classification'
FUNCTION_NAME_REGRESSION = 'regression'
FUNCTION_NAME_CLUSTERING = 'clustering'
FUNCTION_NAME_UNKNOWN = 'unknown'

SUPPORTED_SERIALIZATIONS = ('pickle', 'joblib', 'spark', 'hdf5', 'xgboost', 'lightgbm', 'pmml')


class BaseModel(object):
    def __init__(self, model):
        self.model = model

    def is_support(self):
        raise NotImplementedError()

    def model_type(self):
        raise NotImplementedError()

    def model_version(self):
        raise NotImplementedError()

    def mining_function(self):
        return FUNCTION_NAME_UNKNOWN

    def serialization(self):
        raise NotImplementedError()

    def runtime(self):
        return 'Python{major}{minor}'.format(major=sys.version_info[0], minor=sys.version_info[1])

    def algorithm(self):
        return self.model.__class__.__name__

    def evaluate_metrics(self, x_test, y_test, test_data, input_function_name):
        raise NotImplementedError()

    def predictors(self, x_test, data_test):
        if x_test is None:
            return []

        row = json.loads(x_test.iloc[0].to_json())
        cols = row.keys()
        result = []
        for x in cols:
            result.append(({
                'name': x,
                'sample': row[x],
                'type': type(row[x]).__name__
            }))
        return result

    def targets(self, y_test, data_test):
        if y_test is None:
            return []

        row = json.loads(y_test.iloc[0].to_json())
        cols = row.keys()
        result = []
        for x in cols:
            result.append(({
                'name': x,
                'sample': row[x],
                'type': type(row[x]).__name__
            }))
        return result

    @staticmethod
    def extract_major_minor_version(version):
        result = version
        elements = version.split('.')
        if len(elements) > 2:
            result = '{major}.{minor}'.format(major=elements[0], minor=elements[1])
        return result

    @staticmethod
    def evaluate_metrics_by_sklearn(wrapped_model, x_test, y_test, input_function_name):
        if x_test is None or y_test is None:
            return {}

        try:
            function_name = input_function_name if input_function_name else wrapped_model.mining_function()
            if function_name == FUNCTION_NAME_CLASSIFICATION:
                from sklearn.metrics import accuracy_score
                y_pred = wrapped_model.model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                return {
                 'accuracy': accuracy
                }
            elif function_name == FUNCTION_NAME_REGRESSION:
                from sklearn.metrics import explained_variance_score
                y_pred = wrapped_model.model.predict(x_test)
                explained_variance = explained_variance_score(y_test, y_pred)
                return {
                    'explainedVariance': explained_variance
                }
            else:
                return {}
        except:
            return {}


class CustomModel(BaseModel):
    def __init__(self, model):
        BaseModel.__init__(self, model)

    def is_support(self):
        return not isinstance(self.model, (str, bytes, bytearray))

    def model_type(self):
        return 'Custom'

    def model_version(self):
        return 'unknown'

    def serialization(self):
        return 'pickle'

    def evaluate_metrics(self, x_test, y_test, test_data, input_function_name):
        return {}


class PMMLModel(BaseModel):
    def __init__(self, model):
        self.pmml_model = None
        BaseModel.__init__(self, model)

    def __del__(self):
        if self.pmml_model:
            try:
                from pypmml import Model
                Model.close()
            except:
                pass

    def is_support(self):
        try:
            from pypmml import Model

            model_content = self.model
            if isinstance(model_content, (bytes, bytearray)):
                model_content = model_content.decode('utf-8')

            if isinstance(model_content, str):
                # Check if a file path
                if os.path.exists(model_content):
                    self.pmml_model = Model.fromFile(model_content)
                else:
                    self.pmml_model = Model.fromString(model_content)
                return True
            else:
                Model.close()
                return False
        except Exception as e:
            return False

    def model_type(self):
        return 'PMML'

    def model_version(self):
        return None

    def mining_function(self):
        return self.pmml_model.functionName

    def serialization(self):
        return 'pmml'

    def runtime(self):
        return 'PyPMML'

    def algorithm(self):
        return self.pmml_model.modelElement

    def evaluate_metrics(self, x_test, y_test, test_data, input_function_name):
        prediction_col = self.get_prediction_col()
        if prediction_col is None:
            return {}

        # Convert spark df to Pandas
        if test_data is not None:
            try:
                label_col = self.pmml_model.targetName
                if not label_col:
                    return {}

                pandas_test_data = test_data.toPandas()
                y_test = pandas_test_data[label_col]
                x_test = pandas_test_data
            except:
                return {}

        if x_test is not None and y_test is not None:
            try:
                function_name = input_function_name if input_function_name else self.mining_function()
                if function_name == FUNCTION_NAME_CLASSIFICATION:
                    from sklearn.metrics import accuracy_score
                    y_pred = self.pmml_model.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred[prediction_col])
                    return {
                        'accuracy': accuracy
                    }
                elif function_name == FUNCTION_NAME_REGRESSION:
                    from sklearn.metrics import explained_variance_score
                    y_pred = self.pmml_model.predict(x_test)
                    explained_variance = explained_variance_score(y_test, y_pred[prediction_col])
                    return {
                        'explainedVariance': explained_variance
                    }
                else:
                    return {}
            except:
                return {}
        return {}

    def get_prediction_col(self):
        output_fields = self.pmml_model.outputFields
        for x in output_fields:
            if x.feature == 'predictedValue':
                return x.name
        return None

    def predictors(self, x_test, data_test):
        result = []
        row = None
        if x_test is not None:
            row = json.loads(x_test.iloc[0].to_json())
        elif data_test is not None:
            row = json.loads(data_test.limit(1).toPandas().iloc[0].to_json())

        if row is not None:
            for x in self.pmml_model.inputFields:
                result.append(({
                    'name': x.name,
                    'sample': row.get(x.name),
                    'type': x.dataType
                }))
        return result

    def targets(self, y_test, data_test):
        result = []
        row = None
        if y_test is not None:
            row = json.loads(y_test.iloc[0].to_json())
        elif data_test is not None:
            row = json.loads(data_test.limit(1).toPandas().iloc[0].to_json())

        if row is not None:
            for x in self.pmml_model.targetFields:
                result.append(({
                    'name': x.name,
                    'sample': row.get(x.name),
                    'type': x.dataType
                }))
        return result


class SKLearnModel(BaseModel):
    def __init__(self, model):
        BaseModel.__init__(self, model)

    def is_support(self):
        try:
            from sklearn.base import BaseEstimator
            return isinstance(self.model, BaseEstimator)
        except:
            return False

    def model_type(self):
        return 'Scikit-learn'

    def model_version(self):
        import sklearn
        return BaseModel.extract_major_minor_version(sklearn.__version__)

    def mining_function(self):
        from sklearn.base import is_classifier, is_regressor
        if is_classifier(self.model):
            return FUNCTION_NAME_CLASSIFICATION
        if is_regressor(self.model):
            return FUNCTION_NAME_REGRESSION
        if getattr(self.model, "_estimator_type", None) == "clusterer":
            return FUNCTION_NAME_CLUSTERING
        return BaseModel.mining_function(self)

    def serialization(self):
        return 'joblib'

    def evaluate_metrics(self, x_test, y_test, test_data, input_function_name):
        return BaseModel.evaluate_metrics_by_sklearn(self, x_test, y_test, input_function_name)


class XGBoostModel(BaseModel):
    def __init__(self, model):
        BaseModel.__init__(self, model)

    def is_support(self):
        try:
            import xgboost as xgb
            return isinstance(self.model, xgb.Booster) or \
                   isinstance(self.model, xgb.XGBClassifier) or \
                   isinstance(self.model, xgb.XGBRegressor)
        except:
            return False

    def is_sklearn_format(self):
        import xgboost as xgb
        return isinstance(self.model, xgb.XGBClassifier) or isinstance(self.model, xgb.XGBRegressor)

    def model_type(self):
        return 'XGBoost'

    def model_version(self):
        import xgboost as xgb
        return BaseModel.extract_major_minor_version(xgb.__version__)

    def mining_function(self):
        import xgboost as xgb
        if isinstance(self.model, xgb.XGBClassifier):
            return FUNCTION_NAME_CLASSIFICATION
        if isinstance(self.model, xgb.XGBRegressor):
            return FUNCTION_NAME_REGRESSION
        return BaseModel.mining_function(self)

    def serialization(self):
        return 'joblib' if self.is_sklearn_format() else 'xgboost'

    def evaluate_metrics(self, x_test, y_test, test_data, input_function_name):
        if x_test is None or y_test is None:
            return {}

        if self.is_sklearn_format():
            return BaseModel.evaluate_metrics_by_sklearn(self, x_test, y_test, input_function_name)

        try:
            import xgboost as xgb
            import pandas as pd
            import numpy as np
            function_name = input_function_name if input_function_name else self.mining_function()
            if function_name == FUNCTION_NAME_CLASSIFICATION:
                from sklearn.metrics import accuracy_score
                y_pred = pd.DataFrame(self.model.predict(xgb.DMatrix(x_test))).apply(lambda x: np.argmax(np.array([x])),
                                                                                     axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                return {
                    'accuracy': accuracy
                }
            elif function_name == FUNCTION_NAME_REGRESSION:
                from sklearn.metrics import explained_variance_score
                y_pred = pd.DataFrame(self.model.predict(xgb.DMatrix(x_test)))
                explained_variance = explained_variance_score(y_test, y_pred)
                return {
                    'explainedVariance': explained_variance
                }
            else:
                return {}
        except:
            return {}


class LightGBMModel(BaseModel):
    def __init__(self, model):
        BaseModel.__init__(self, model)

    def is_support(self):
        try:
            import lightgbm as lgb
            return isinstance(self.model, lgb.Booster) or \
                   isinstance(self.model, lgb.LGBMClassifier) or \
                   isinstance(self.model, lgb.LGBMRegressor)
        except:
            return False

    def is_sklearn_format(self):
        import lightgbm as lgb
        return isinstance(self.model, lgb.LGBMClassifier) or isinstance(self.model, lgb.LGBMRegressor)

    def model_type(self):
        return 'LightGBM'

    def model_version(self):
        import lightgbm as lgb
        return BaseModel.extract_major_minor_version(lgb.__version__)

    def mining_function(self):
        import lightgbm as lgb
        if isinstance(self.model, lgb.LGBMClassifier):
            return FUNCTION_NAME_CLASSIFICATION
        if isinstance(self.model, lgb.LGBMRegressor):
            return FUNCTION_NAME_REGRESSION
        return BaseModel.mining_function(self)

    def serialization(self):
        return 'joblib' if self.is_sklearn_format() else 'lightgbm'

    def evaluate_metrics(self, x_test, y_test, test_data, input_function_name):
        if x_test is None or y_test is None:
            return {}

        if self.is_sklearn_format():
            return BaseModel.evaluate_metrics_by_sklearn(self, x_test, y_test, input_function_name)

        try:
            import lightgbm as lgb
            import pandas as pd
            import numpy as np
            function_name = input_function_name if input_function_name else self.mining_function()
            if function_name == FUNCTION_NAME_CLASSIFICATION:
                from sklearn.metrics import accuracy_score
                y_pred = pd.DataFrame(self.model.predict(x_test)).apply(lambda x: np.argmax(np.array([x])),
                                                                                     axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                return {
                    'accuracy': accuracy
                }
            elif function_name == FUNCTION_NAME_REGRESSION:
                from sklearn.metrics import explained_variance_score
                y_pred = pd.DataFrame(self.model.predict(x_test))
                explained_variance = explained_variance_score(y_test, y_pred)
                return {
                    'explainedVariance': explained_variance
                }
            else:
                return {}
        except:
            return {}


class KerasModel(BaseModel):
    def __init__(self, model):
        BaseModel.__init__(self, model)

    def is_support(self):
        try:
            from keras.models import Model
            return isinstance(self.model, Model)
        except:
            return False

    def model_type(self):
        return 'Keras'

    def model_version(self):
        import keras
        return BaseModel.extract_major_minor_version(keras.__version__)

    def mining_function(self):
        return BaseModel.mining_function(self)

    def serialization(self):
        return 'hdf5'

    def evaluate_metrics(self, x_test, y_test, test_data, input_function_name):
        if x_test is None or y_test is None:
            return {}

        try:
            import numpy as np
            import pandas as pd
            function_name = input_function_name if input_function_name else self.mining_function()

            shape = y_test.shape
            if len(shape) > 1 and shape[1] > 1:
                y_test = np.argmax(y_test.values, axis=1)

            if function_name == FUNCTION_NAME_CLASSIFICATION:
                from sklearn.metrics import accuracy_score
                y_pred = pd.DataFrame(self.model.predict(x_test.as_matrix())).apply(lambda x: np.argmax(np.array([x])),
                                                                                     axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                return {
                    'accuracy': accuracy
                }
            elif function_name == FUNCTION_NAME_REGRESSION:
                from sklearn.metrics import explained_variance_score
                y_pred = pd.DataFrame(self.model.predict(x_test.as_matrix()))
                explained_variance = explained_variance_score(y_test, y_pred)
                return {
                    'explainedVariance': explained_variance
                }
            else:
                return {}
        except:
            return {}


class SparkModel(BaseModel):
    def __init__(self, model):
        BaseModel.__init__(self, model)

    def is_support(self):
        try:
            from pyspark.ml import Model
            return isinstance(self.model, Model)
        except:
            return False

    def is_pipeline_model(self):
        try:
            from pyspark.ml import PipelineModel
            return isinstance(self.model, PipelineModel)
        except:
            return False

    def model_type(self):
        return 'Spark'

    def model_version(self):
        from pyspark import SparkConf, SparkContext
        sc = SparkContext.getOrCreate(conf=SparkConf())
        return BaseModel.extract_major_minor_version(sc.version)

    def mining_function(self):
        return BaseModel.mining_function(self)

    def serialization(self):
        return 'spark'

    def evaluate_metrics(self, x_test, y_test, test_data, input_function_name):
        if test_data is None:
            return {}

        try:
            prediction = self.model.transform(test_data)
            label_col = self.get_label_col()
            predict_col = self.get_prediction_col()
            function_name = input_function_name if input_function_name else self.mining_function()
            if function_name == FUNCTION_NAME_CLASSIFICATION:
                accuracy = prediction.rdd.filter(
                    lambda x: x[label_col] == x[predict_col]).count() * 1.0 / prediction.count()
                return {
                    'accuracy': accuracy
                }
            elif function_name == FUNCTION_NAME_REGRESSION:
                numerator = prediction.rdd.map(lambda x: x[label_col] - x[predict_col]).variance()
                denominator = prediction.rdd.map(lambda x: x[label_col]).variance()
                explained_variance = 1.0 - numerator / denominator
                return {
                    'explainedVariance': explained_variance
                }
            else:
                return {}
        except:
            return {}

    def predictors(self, x_test, data_test):
        if data_test is None:
            return []

        row = json.loads(data_test.limit(1).toPandas().iloc[0].to_json())
        label_col = self.get_label_col()
        cols = row.keys()
        result = []
        for x in cols:
            if x != label_col:
                result.append(({
                    'name': x,
                    'sample': row[x],
                    'type': type(row[x]).__name__
                }))
        return result

    def targets(self, y_test, data_test):
        if data_test is None:
            return []

        row = json.loads(data_test.limit(1).toPandas().iloc[0].to_json())
        label_col = self.get_label_col()
        cols = row.keys()
        result = []
        for x in cols:
            if x == label_col:
                result.append(({
                    'name': x,
                    'sample': row[x],
                    'type': type(row[x]).__name__
                }))
        return result

    def get_label_col(self):
        from pyspark.ml import PipelineModel
        if isinstance(self.model, PipelineModel):
            stages = self.model.stages
            label_col = None
            i = 0
            for x in reversed(stages):
                try:
                    label_col = x._call_java('getLabelCol')
                    i += 1
                    break;
                except:
                    pass

            # find the first input column
            reversed_stages = stages[:]
            reversed_stages.reverse()
            for x in reversed_stages[i:]:
                try:
                    if x._call_java('getOutputCol') == label_col:
                        label_col = x._call_java('getInputCol')
                except:
                    pass
            return 'label' if label_col is None else label_col
        else:
            label_col = None
            try:
                label_col = self.model._call_java('getLabelCol')
            except:
                label_col = 'label'
            return label_col

    def get_prediction_col(self):
        from pyspark.ml import PipelineModel
        if isinstance(self.model, PipelineModel):
            stages = self.model.stages
            try:
                return stages[-1].getOutputCol()
            except:
                return 'prediction'
        else:
            try:
                return self.model.getPredictionCol()
            except:
                return 'prediction'


def get_model_metadata(model,
                       mining_function=None,
                       X_test=None,
                       y_test=None,
                       data_test=None,
                       features_json=None,
                       labels_json=None):
    # The order of such list is significant, do not change it!
    candidates = [LightGBMModel, XGBoostModel, SKLearnModel, SparkModel, KerasModel, PMMLModel, CustomModel]

    wrapped_model = None
    for cls in candidates:
        wrapped_model = cls(model)
        if wrapped_model.is_support():
            break
        else:
            wrapped_model = None

    if wrapped_model is None:
        raise ValueError('The model {class_name} is not recognized.'.format(class_name=model.__class__.__name__))

    if wrapped_model.model_type() == 'Spark' and not wrapped_model.is_pipeline_model():
        raise ValueError("The Spark model should be a PipelineModel, %s was given" % wrapped_model.__class__.__name__)

    X_test = pd.DataFrame(X_test) if X_test is not None and not isinstance(X_test, pd.DataFrame) else X_test
    y_test = pd.DataFrame(y_test) if y_test is not None and not isinstance(y_test, pd.DataFrame) else y_test

    return {
        'runtime': wrapped_model.runtime(),
        'type': wrapped_model.model_type(),
        'appVersion': wrapped_model.model_version(),
        'functionName': mining_function if mining_function else wrapped_model.mining_function(),
        'serialization': wrapped_model.serialization(),
        'algorithm': wrapped_model.algorithm(),
        'metrics': wrapped_model.evaluate_metrics(X_test, y_test, data_test, mining_function),
        'predictors': features_json if features_json else wrapped_model.predictors(X_test, data_test),
        'targets': labels_json if labels_json else wrapped_model.targets(y_test, data_test)
    }


def save_model(model, model_path, serialization=None):
    if serialization is None:
        metadata = get_model_metadata(model)
        serialization = metadata['serialization']

    if serialization not in SUPPORTED_SERIALIZATIONS:  # pragma: no cover
        raise ValueError("serialization should be one of %s, %s was given" % (
            SUPPORTED_SERIALIZATIONS, serialization))

    if serialization == 'joblib':
        from sklearn.externals import joblib
        joblib.dump(model, model_path)
    elif serialization == 'pickle':
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    elif serialization == 'xgboost':
        model.save_model(model_path)
    elif serialization == 'hdf5':
        model.save(model_path)
    elif serialization == 'spark':
        from pyspark.ml import PipelineModel
        model.write().overwrite().save(model_path)
    elif serialization == 'pmml':
        if os.path.exists(model):
            with open(model, mode='rb') as f:
                model = f.read()
        mode = 'wb' if isinstance(model, (bytes, bytearray)) else 'w'
        with open(model_path, mode) as file:
            file.write(model)
    elif serialization == 'lightgbm':
        model.save_model(model_path)

    return model
