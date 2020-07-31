# DaaS-Client

_DaaS-Client_ is Python client library for DaaS(Deployment-as-a-Service)

## Features
_DaaS-Client_ helps to publish your AI/ML models in Python, test and deploy them easily.

It supports the following models by default, more types will be added in the list.
* Scikit-learn
* XGBoost
* LightGBM
* Keras and Tensorflow(tf.keras)
* PySpark
* PMML
* ONNX
* PyTorch
* Custom models

## Prerequisites
 - Python 2.7 or >= 3.5

## Dependencies
  - requests
  - numpy
  - pandas
  - pypmml
  - onnx
  - onnxruntime
  
## Installation

Install the latest version from github:

```bash
pip install --upgrade git+https://github.com/autodeployai/daas-client.git
```

## Usage
1. Initiate a client with URL of DaaS server, username, password, and optional project, e.g.

    ```python
    from daas_client import DaasClient
    
    client = DaasClient('https://192.168.64.3:31753', 'admin', 'password')
    ```

2. Call `publish` to publish models into DaaS server. There are two methods to call this function, one is for PySpark, the other is for others. For the PMML model, you can use either.

    Load the iris data from sklearn datasets
    ```python
    iris = datasets.load_iris()
    iris_target_name = 'Species'
    iris_feature_names = iris.feature_names
    iris_df = pd.DataFrame(iris.data, columns=iris_feature_names)
    iris_df[iris_target_name] = iris.target
    ```

    Train and publish a Random Forest model of PySpark. NOTE: the spark model must be a Pipeline model
    ```python
    df = spark.createDataFrame(iris_df)
    df_train, df_test = df.randomSplit([0.7, 0.3])
    assembler = VectorAssembler(inputCols=iris_feature_names, outputCol='features')
    rf = RandomForestClassifier().setLabelCol(iris_target_name)
    pipe = Pipeline(stages=[assembler, rf])
    model = pipe.fit(df_train)

    publish_resp = client.publish(model, name='spark-cls', mining_function='classification', data_test=df_test, description='A Spark classification model')
    ```

    Train and publish a XGBoost model
    ```python
    X, y = iris_df[iris_feature_names], iris_df[iris_target_name]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = XGBClassifier(max_depth=3, objective='multi:softprob')
    model.fit(x_train, y_train)

    publish_resp = client.publish(model, name='xgboost-cls', mining_function='classification', x_test=x_test, y_test=y_test, description='A XGBoost classification model')
    ```
    
    The result is a dict with published model name and version if success, e.g.
    ```python
    {'model_name': 'xgboost-cls', 'model_version': '1'}
    ```

3. Call `test` to test the published model in the development mode.

    ```python
    test_resp = client.test('xgboost-cls', model_version=publish_resp['model_version'])
    ```
    
    The result is a dict with all info of the REST service of testing a published model with its version, e.g.
    ```python
    {'access_token': 'A-LONG-STRING-OF-BEARER-TOKEN-USED-IN-HTTP-HEADER-AUTHORIZATION',
     'endpoint_url': 'https://192.168.64.3:31753/api/v1/test/examples/daas-python37-faas/test',
     'payload': {'args': {'X': [{'petal length (cm)': 1.5,
                                 'petal width (cm)': 0.4,
                                 'sepal length (cm)': 5.7,
                                 'sepal width (cm)': 4.4}],
                          'model_name': 'xgboost-cls',
                          'model_version': '1'}}}
        
    ```

4. Call `deploy` to deploy the published model in the product mode.

    ```python
    deploy_resp = client.deploy('xgboost-cls', deployment_name='xgboost-cls-svc', model_version=publish_resp['model_version'])
    ```
    
    The result is a dict with all info of the REST service of deploying a published model with its version, e.g.
    ```python
    {'access_token': 'A-LONG-STRING-OF-BEARER-TOKEN-USED-IN-HTTP-HEADER-AUTHORIZATION',
     'endpoint_url': 'https://192.168.64.3:31753/api/v1/svc/examples/xgboost-cls-svc/predict',
     'payload': {'args': {'X': [{'petal length (cm)': 1.5,
                                 'petal width (cm)': 0.4,
                                 'sepal length (cm)': 5.7,
                                 'sepal width (cm)': 4.4}]}}}
    ```

You can refer to the example Jupyter notebooks for more details.


## Support
If you have any questions about the _DaaS-Client_ library, please open issues on this repository.

## License
_DaaS-Client_ is licensed under [APL 2.0](http://www.apache.org/licenses/LICENSE-2.0).
