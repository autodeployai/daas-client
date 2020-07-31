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

from __future__ import absolute_import, print_function
import sys
import json
import os
import pandas as pd
import numpy as np

FUNCTION_NAME_CLASSIFICATION = 'classification'
FUNCTION_NAME_REGRESSION = 'regression'
FUNCTION_NAME_CLUSTERING = 'clustering'
FUNCTION_NAME_UNKNOWN = 'unknown'

SUPPORTED_FUNCTION_NAMES = (FUNCTION_NAME_CLASSIFICATION, FUNCTION_NAME_REGRESSION, FUNCTION_NAME_CLUSTERING)
SUPPORTED_SERIALIZATIONS = ('pickle', 'joblib', 'spark', 'hdf5', 'xgboost', 'lightgbm', 'pmml', 'onnx', 'pt')


class BaseModel(object):
    def __init__(self, model):
        self.model = model

    def is_support(self):
        raise NotImplementedError()

    def model_type(self):
        raise NotImplementedError()

    def model_version(self):
        raise NotImplementedError()

    def mining_function(self, y_test):
        return FUNCTION_NAME_UNKNOWN

    def serialization(self):
        raise NotImplementedError()

    def runtime(self):
        return 'Python{major}{minor}'.format(major=sys.version_info[0], minor=sys.version_info[1])

    def algorithm(self):
        return self.model.__class__.__name__

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
        raise NotImplementedError()

    def predictors(self, x_test, data_test):
        if x_test is None:
            return []

        result = []
        if isinstance(x_test, np.ndarray) and x_test.ndim <= 2:
            x_test = pd.DataFrame(x_test)
            x_test.columns = ['x'+str(i) for i in range(0, len(x_test.columns))]

        x_test = self._series_to_dataframe(x_test)
        if isinstance(x_test, pd.DataFrame):
            row = json.loads(x_test.iloc[0].to_json())
            cols = row.keys()
            for x in cols:
                result.append({
                    'name': x,
                    'sample': row[x],
                    'type': type(row[x]).__name__
                })
        else:  # numpy array with multiple dimensions than two
            row = x_test[0]
            result.append({
                'name': 'tensor_input',
                'sample': row.tolist(),
                'type': x_test.dtype.name,
                'shape': self._normalize_np_shape(x_test.shape)
            })

        return result

    def targets(self, y_test, data_test):
        if y_test is None:
            return []

        result = []
        if isinstance(y_test, np.ndarray) and y_test.ndim <= 2:
            y_test = pd.DataFrame(y_test)
            y_test.columns = ['y'+str(i) for i in range(0, len(y_test.columns))]

        y_test = self._series_to_dataframe(y_test)
        if isinstance(y_test, pd.DataFrame):
            row = json.loads(y_test.iloc[0].to_json())
            cols = row.keys()
            for x in cols:
                result.append({
                    'name': x,
                    'sample': row[x],
                    'type': type(row[x]).__name__
                })
        else:  # numpy array with multiple dimensions than two
            row = y_test[0]
            result.append({
                'name': 'tensor_target',
                'sample': row.tolist(),
                'type': y_test.dtype.name,
                'shape': self._normalize_np_shape(y_test.shape)
            })

        return result

    def outputs(self, y_test, data_test, **kwargs):
        return []

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
            function_name = input_function_name if input_function_name else wrapped_model.mining_function(y_test)
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

    @staticmethod
    def _normalize_np_shape(shape):
        result = None
        if shape is not None and len(shape) > 1:
            result = []
            for idx, d in enumerate(shape):
                if idx == 0:
                    result.append(None)
                else:
                    result.append(d)
        return result

    @staticmethod
    def _series_to_dataframe(data):
        if isinstance(data, pd.Series):
            return pd.DataFrame(data)
        return data

    def _test_data_to_ndarray(self, x_y_test, data_test):
        data = self._to_dataframe(x_y_test, data_test)
        if isinstance(data, pd.DataFrame):
            return data.values
        return data

    @staticmethod
    def _to_ndarray(data):
        return data.values if isinstance(data, (pd.DataFrame, pd.Series)) else data

    @staticmethod
    def _to_dataframe(x_y_test, data_test):
        if x_y_test is None and data_test is not None:
            x_y_test = data_test.limit(1).toPandas()
        if isinstance(x_y_test, pd.Series):
            x_y_test = pd.DataFrame(x_y_test)
        return x_y_test

    def _infer_mining_function(self, y_test):
        if y_test is None:
            return FUNCTION_NAME_UNKNOWN

        y_test = self._to_ndarray(y_test)
        if y_test.ndim >= 2:
            return FUNCTION_NAME_CLASSIFICATION if y_test.shape[y_test.ndim - 1] > 1 else FUNCTION_NAME_REGRESSION

        # float numbers are treated as a regression problem
        return FUNCTION_NAME_REGRESSION if y_test.dtype.kind in 'fc' else FUNCTION_NAME_CLASSIFICATION

    @staticmethod
    def _compatible_shape(shape1, shape2):
        if len(shape1) != len(shape2):
            return False

        # could be tuple and list
        shape1 = list(shape1)
        shape2 = list(shape2)
        if len(shape1) > 1:
            return shape1[1:] == shape2[1:]
        return shape1 == shape2


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

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
        return {}


class PMMLModel(BaseModel):
    def __init__(self, model):
        BaseModel.__init__(self, model)
        self.pmml_model = None

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
            if hasattr(self.model, 'read') and callable(self.model.read):
                model_content = self.model.read()

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

    def mining_function(self, y_test):
        return self.pmml_model.functionName

    def serialization(self):
        return 'pmml'

    def runtime(self):
        return 'PyPMML'

    def algorithm(self):
        return self.pmml_model.modelElement

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
        prediction_col = self.get_prediction_col()
        if prediction_col is None:
            return {}

        # Convert spark df to Pandas
        if data_test is not None:
            try:
                label_col = self.pmml_model.targetName
                if not label_col:
                    return {}

                pandas_data_test = data_test.toPandas()
                y_test = pandas_data_test[label_col]
                x_test = pandas_data_test
            except:
                return {}

        if x_test is not None and y_test is not None:
            try:
                function_name = input_function_name if input_function_name else self.mining_function(y_test)
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
        x_test = self._to_dataframe(x_test, data_test)
        if isinstance(x_test, pd.DataFrame):
            row = json.loads(x_test.iloc[0].to_json())

        for x in self.pmml_model.inputFields:
            result.append(({
                'name': x.name,
                'sample': row.get(x.name) if row is not None else None,
                'type': x.dataType
            }))
        return result

    def targets(self, y_test, data_test):
        result = []

        row = None
        y_test = self._to_dataframe(y_test, data_test)
        if isinstance(y_test, pd.DataFrame):
            row = json.loads(y_test.iloc[0].to_json())

        for x in self.pmml_model.targetFields:
            result.append(({
                'name': x.name,
                'sample': row.get(x.name) if row is not None else None,
                'type': x.dataType
            }))
        return result

    def outputs(self, y_test, data_test, **kwargs):
        result = []
        for x in self.pmml_model.outputFields:
            result.append(({
                'name': x.name,
                'type': x.dataType
            }))
        return result


class ONNXModel(BaseModel):
    def __init__(self, model):
        super(ONNXModel, self).__init__(model)
        self.onnx_model = None
        self.sess = None
        self._algorithm = None

    def is_support(self):
        try:
            import onnx

            if isinstance(self.model, onnx.ModelProto):
                self.onnx_model = self.model
                return True

            if isinstance(self.model, (bytes, bytearray)):
                onnx_model = onnx.load_model_from_string(self.model)
            else:
                # could be either readable or a file path
                onnx_model = onnx.load_model(self.model)

            onnx.checker.check_model(onnx_model)
            self.onnx_model = onnx_model
            return True
        except Exception:
            return False

    def model_type(self):
        return 'ONNX'

    def model_version(self):
        return None

    def mining_function(self, y_test):
        algorithm = self.algorithm()
        if algorithm is not None:
            if algorithm in ('LinearClassifier', 'SVMClassifier', 'TreeEnsembleClassifier'):
                return FUNCTION_NAME_CLASSIFICATION
            if algorithm in ('LinearRegressor', 'SVMRegressor', 'TreeEnsembleRegressor'):
                return FUNCTION_NAME_REGRESSION
        return self._infer_mining_function(y_test)

    def serialization(self):
        return 'onnx'

    def runtime(self):
        return 'ONNX Runtime'

    def algorithm(self):
        if self._algorithm is None:
            use_onnx_ml = False
            if self.onnx_model is not None:
                graph = self.onnx_model.graph
                for node in graph.node:
                    if node.domain == 'ai.onnx.ml':
                        use_onnx_ml = True
                        if node.op_type in ('LinearClassifier', 'LinearRegressor', 'SVMClassifier', 'SVMRegressor',
                                            'TreeEnsembleClassifier', 'TreeEnsembleRegressor'):
                            self._algorithm = node.op_type
                            break
                if self._algorithm is None and not use_onnx_ml:
                    self._algorithm = 'NeuralNetwork'
        return self._algorithm

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
        if x_test is None or y_test is None:
            return {}

        try:
            function_name = input_function_name if input_function_name else self.mining_function(y_test)

            # convert to numpy array if not
            x_test = self._to_ndarray(x_test)
            y_test = self._to_ndarray(y_test)

            shape = y_test.shape
            if len(shape) > 1 and shape[1] > 1:
                y_test = np.argmax(y_test, axis=1)

            sess = self._get_inference_session()
            y_pred = None
            if function_name in (FUNCTION_NAME_CLASSIFICATION, FUNCTION_NAME_REGRESSION) and len(
                    sess.get_inputs()) == 1:
                input_name = sess.get_inputs()[0].name
                y_pred = sess.run([sess.get_outputs()[0].name], {input_name: x_test.astype(np.float32)})[0]
                y_pred = np.asarray(y_pred)
                shape = y_pred.shape
                if len(shape) > 1 and shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)

            if y_pred is not None:
                if function_name == FUNCTION_NAME_CLASSIFICATION:
                    from sklearn.metrics import accuracy_score
                    accuracy = accuracy_score(y_test, y_pred)
                    return {
                        'accuracy': accuracy
                    }
                elif function_name == FUNCTION_NAME_REGRESSION:
                    from sklearn.metrics import explained_variance_score
                    explained_variance = explained_variance_score(y_test, y_pred)
                    return {
                        'explainedVariance': explained_variance
                    }
            else:
                return {}
        except Exception as e:
            return {}

    def predictors(self, x_test, data_test):
        result = []

        sess = self._get_inference_session()
        for x in sess.get_inputs():
            result.append({
                'name': x.name,
                'type': x.type,
                'shape': x.shape
            })

        # suppose there is only 1 tensor input
        data = self._test_data_to_ndarray(x_test, data_test)
        if data is not None and len(result) == 1:
            if self._compatible_shape(data.shape, result[0]['shape']):
                result[0]['sample'] = [data[0].tolist()]
        return result

    def targets(self, y_test, data_test):
        return []

    def outputs(self, y_test, data_test, **kwargs):
        result = []

        sess = self._get_inference_session()
        for x in sess.get_outputs():
            result.append({
                'name': x.name,
                'type': x.type,
                'shape': x.shape
            })
        return result

    def _get_inference_session(self):
        if self.sess is None:
            import onnxruntime as rt
            self.sess = rt.InferenceSession(self.onnx_model.SerializeToString())
        return self.sess


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

    def mining_function(self, y_test):
        from sklearn.base import is_classifier, is_regressor
        if is_classifier(self.model):
            return FUNCTION_NAME_CLASSIFICATION
        if is_regressor(self.model):
            return FUNCTION_NAME_REGRESSION
        if getattr(self.model, "_estimator_type", None) == "clusterer":
            return FUNCTION_NAME_CLUSTERING
        return self._infer_mining_function(y_test)

    def serialization(self):
        return 'joblib'

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
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

    def mining_function(self, y_test):
        import xgboost as xgb
        if isinstance(self.model, xgb.XGBClassifier):
            return FUNCTION_NAME_CLASSIFICATION
        if isinstance(self.model, xgb.XGBRegressor):
            return FUNCTION_NAME_REGRESSION
        return self._infer_mining_function(y_test)

    def serialization(self):
        return 'joblib' if self.is_sklearn_format() else 'xgboost'

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
        if x_test is None or y_test is None:
            return {}

        if self.is_sklearn_format():
            return BaseModel.evaluate_metrics_by_sklearn(self, x_test, y_test, input_function_name)

        try:
            import xgboost as xgb
            import pandas as pd
            import numpy as np
            function_name = input_function_name if input_function_name else self.mining_function(y_test)
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

    def mining_function(self, y_test):
        import lightgbm as lgb
        if isinstance(self.model, lgb.LGBMClassifier):
            return FUNCTION_NAME_CLASSIFICATION
        if isinstance(self.model, lgb.LGBMRegressor):
            return FUNCTION_NAME_REGRESSION
        return self._infer_mining_function(y_test)

    def serialization(self):
        return 'joblib' if self.is_sklearn_format() else 'lightgbm'

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
        if x_test is None or y_test is None:
            return {}

        if self.is_sklearn_format():
            return BaseModel.evaluate_metrics_by_sklearn(self, x_test, y_test, input_function_name)

        try:
            import lightgbm as lgb
            import pandas as pd
            import numpy as np
            function_name = input_function_name if input_function_name else self.mining_function(y_test)
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
        self.tf_keras = False

    def is_support(self):
        try:
            from keras.models import Model
            if isinstance(self.model, Model):
                return True
            return self._is_support_tf_keras()
        except:
            return self._is_support_tf_keras()
            
    def _is_support_tf_keras(self):
        try:
            import tensorflow as tf
            self.tf_keras = isinstance(self.model, tf.keras.Model)
            return self.tf_keras
        except:
            return False

    def model_type(self):
        return 'tf.Keras' if self.tf_keras else 'Keras'

    def model_version(self):
        if self.tf_keras:
            import tensorflow as tf
            return BaseModel.extract_major_minor_version(tf.keras.__version__)
        else:
            import keras
            return BaseModel.extract_major_minor_version(keras.__version__)

    def mining_function(self, y_test):
        return self._infer_mining_function(y_test)

    def serialization(self):
        return 'hdf5'

    def predictors(self, x_test, data_test):
        result = []

        row = None
        columns = None
        if x_test is not None:
            x_test = self._series_to_dataframe(x_test)
            shape = x_test.shape
            if isinstance(x_test, pd.DataFrame):
                row = x_test.iloc[0]
                columns = list(x_test.columns)
            else:
                row = x_test[0]

        for idx, x in enumerate(self.model.inputs):
            name = x.name
            if hasattr(self.model, 'input_names'):
                name = self.model.input_names[idx]
            tensor_shape = self._normalize_tensor_shape(x.shape)
            result.append({
                'name': name,
                'sample': [row.tolist()] if row is not None and self._compatible_shape(tensor_shape, shape) else None,
                'type': np.dtype(x.dtype.as_numpy_dtype).name,
                'shape': tensor_shape
            })

            if columns is not None and result[-1]['sample'] is not None:
                result[-1]['columns'] = columns

        return result

    def targets(self, y_test, data_test):
        if y_test is None:
            return []

        result = []
        y_test = self._series_to_dataframe(y_test)
        if isinstance(y_test, pd.DataFrame):
            row = json.loads(y_test.iloc[0].to_json())
            cols = row.keys()
            for x in cols:
                result.append(({
                    'name': x,
                    'sample': row[x],
                    'type': type(row[x]).__name__
                }))
        else:
            row = y_test[0]
            result.append({
                'name': 'tensor_target',
                'sample': row.tolist(),
                'type': y_test.dtype.name,
                'shape': self._normalize_np_shape(y_test.shape)
            })

        return result

    def outputs(self, y_test, data_test, **kwargs):
        result = []

        for idx, x in enumerate(self.model.outputs):
            name = x.name
            if hasattr(self.model, 'output_names'):
                name = self.model.output_names[idx]
            result.append(({
                'name': name,
                'type': np.dtype(x.dtype.as_numpy_dtype).name,
                'shape': self._normalize_tensor_shape(x.shape)
            }))
        return result

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
        if x_test is None or y_test is None:
            return {}

        try:
            import numpy as np
            import pandas as pd
            function_name = input_function_name if input_function_name else self.mining_function(y_test)

            # convert to numpy array if not
            x_test = BaseModel._to_ndarray(x_test)
            y_test = BaseModel._to_ndarray(y_test)

            shape = y_test.shape
            if len(shape) > 1 and shape[1] > 1:
                y_test = np.argmax(y_test, axis=1)

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

    @staticmethod
    def _normalize_tensor_shape(tensor_shape):
        return [d.value for d in tensor_shape]


class PytorchModel(BaseModel):
    def __init__(self, model):
        BaseModel.__init__(self, model)

    def is_support(self):
        try:
            from torch import nn
            return isinstance(self.model, nn.Module)
        except:
            return False

    def model_type(self):
        return 'Pytorch'

    def model_version(self):
        import torch
        return BaseModel.extract_major_minor_version(torch.__version__)

    def mining_function(self, y_test):
        return self._infer_mining_function(y_test)

    def serialization(self):
        return 'pt'

    def predictors(self, x_test, data_test):
        result = []

        columns = None
        shape = None
        sample = None
        if x_test is not None:
            x_test = self._series_to_dataframe(x_test)
            dtype = x_test.dtype
            shape = x_test.shape
            if isinstance(x_test, pd.DataFrame):
                row = x_test.iloc[0]
                columns = list(x_test.columns)
            else:
                row = x_test[0]
            sample = [row.tolist()]
        else:
            import torch
            dtype = torch.Tensor(1).numpy().dtype

        result.append({
            'name': 'tensor_input',
            'sample': sample,
            'type': dtype.name,
            'shape': self._normalize_np_shape(shape)
        })

        if columns is not None and result[-1]['sample'] is not None:
            result[-1]['columns'] = columns

        return result

    def targets(self, y_test, data_test):
        if y_test is None:
            return []

        result = []
        y_test = self._series_to_dataframe(y_test)
        if isinstance(y_test, pd.DataFrame):
            row = json.loads(y_test.iloc[0].to_json())
        else:
            row = y_test[0]

        result.append({
            'name': 'tensor_target',
            'sample': row.tolist(),
            'type': y_test.dtype.name,
            'shape': self._normalize_np_shape(y_test.shape)
        })

        return result

    def outputs(self, y_test, data_test, **kwargs):
        result = []

        if 'x_test' in kwargs:
            x_test = self._to_ndarray(kwargs['x_test'])
            if x_test is not None:
                shape = list(x_test.shape)
                if len(shape) > 0 and shape[0] > 1:
                    shape[0] = 1

                import torch
                data = self.model(torch.randn(*shape)).data.numpy()
                result.append(({
                    'name': 'tensor_output',
                    'type': data.dtype.name,
                    'shape': self._normalize_np_shape(data.shape)
                }))
        return result

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
        if x_test is None or y_test is None:
            return {}

        try:
            import numpy as np
            import pandas as pd
            import torch
            function_name = input_function_name if input_function_name else self.mining_function(y_test)

            # convert to numpy array if not
            x_test = BaseModel._to_ndarray(x_test)
            y_test = BaseModel._to_ndarray(y_test)

            shape = y_test.shape
            if len(shape) > 1 and shape[1] > 1:
                y_test = np.argmax(y_test, axis=1)

            if function_name == FUNCTION_NAME_CLASSIFICATION:
                from sklearn.metrics import accuracy_score
                dtype = torch.Tensor(1).dtype
                data = self.model(torch.from_numpy(x_test).type(dtype)).data.numpy()
                y_pred = pd.DataFrame(data).apply(lambda x: np.argmax(np.array([x])), axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                return {
                    'accuracy': accuracy
                }
            elif function_name == FUNCTION_NAME_REGRESSION:
                from sklearn.metrics import explained_variance_score
                dtype = torch.Tensor(1).dtype
                data = self.model(torch.from_numpy(x_test).type(dtype)).data.numpy()
                y_pred = pd.DataFrame(data)
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

    def mining_function(self, y_test):
        return BaseModel.mining_function(self, y_test)

    def serialization(self):
        return 'spark'

    def evaluate_metrics(self, x_test, y_test, data_test, input_function_name):
        if data_test is None:
            return {}

        try:
            prediction = self.model.transform(data_test)
            label_col = self.get_label_col()
            predict_col = self.get_prediction_col()
            function_name = input_function_name if input_function_name else self.mining_function(y_test)
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


def _ndarray_or_dataframe(data):
    if data is not None:
        if isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            return data
        return np.asarray(data)
    return None


def get_model_metadata(model,
                       mining_function=None,
                       x_test=None,
                       y_test=None,
                       data_test=None,
                       features_json=None,
                       labels_json=None,
                       outputs_json=None,
                       source_object=None):
    # The order of such list is significant, do not change it!
    candidates = [LightGBMModel, XGBoostModel, SKLearnModel, SparkModel, KerasModel, PytorchModel,
                  PMMLModel, ONNXModel, CustomModel]

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

    if mining_function is not None and mining_function not in SUPPORTED_FUNCTION_NAMES:
        raise ValueError("mining_function should be one of %s, %s was given" % (
            SUPPORTED_FUNCTION_NAMES, mining_function))

    x_test = _ndarray_or_dataframe(x_test)
    y_test = _ndarray_or_dataframe(y_test)

    # get the source code of an input object
    object_name = None
    object_source = None
    if source_object is not None:
        try:
            import inspect
            object_name = source_object.__name__
            object_source = inspect.getsource(source_object)
        except:
            pass

    return {
        'runtime': wrapped_model.runtime(),
        'type': wrapped_model.model_type(),
        'appVersion': wrapped_model.model_version(),
        'functionName': mining_function if mining_function else wrapped_model.mining_function(y_test),
        'serialization': wrapped_model.serialization(),
        'algorithm': wrapped_model.algorithm(),
        'metrics': wrapped_model.evaluate_metrics(x_test, y_test, data_test, mining_function),
        'predictors': features_json if features_json else wrapped_model.predictors(x_test, data_test),
        'targets': labels_json if labels_json else wrapped_model.targets(y_test, data_test),
        'outputs': outputs_json if outputs_json else wrapped_model.outputs(y_test, data_test, x_test=x_test),
        'objectSource': object_source,
        'objectName': object_name
    }


def save_model(model, model_path, serialization=None):
    if serialization is None:
        metadata = get_model_metadata(model)
        serialization = metadata['serialization']

    if serialization not in SUPPORTED_SERIALIZATIONS:  # pragma: no cover
        raise ValueError("serialization should be one of %s, %s was given" % (
            SUPPORTED_SERIALIZATIONS, serialization))

    raw_model = model
    if serialization == 'joblib':
        try:
            import joblib
        except ImportError:
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
    elif serialization == 'pt':
        import torch
        torch.save(model.state_dict(), model_path)
    elif serialization == 'spark':
        from pyspark.ml import PipelineModel
        model.write().overwrite().save(model_path)
    elif serialization == 'pmml':
        if hasattr(model, 'read') and callable(model.read):
            model = model.read()
        if os.path.exists(model):
            with open(model, mode='rb') as f:
                model = f.read()
        mode = 'wb' if isinstance(model, (bytes, bytearray)) else 'w'
        with open(model_path, mode) as file:
            file.write(model)
    elif serialization == 'onnx':
        import onnx
        if isinstance(model, onnx.ModelProto):
            onnx_model = model
        elif isinstance(model, (bytes, bytearray)):
            onnx_model = onnx.load_model_from_string(model)
        else:
            onnx_model = onnx.load_model(model)
        onnx.save(onnx_model, model_path)
    elif serialization == 'lightgbm':
        model.save_model(model_path)

    return raw_model
