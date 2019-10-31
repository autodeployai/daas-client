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
import requests
import json
import tempfile
import os
import shutil
import time
from daas_client.ml import get_model_metadata, save_model

# disable InsecureRequestWarning
requests.packages.urllib3.disable_warnings()


class ApiException(Exception):
    """Class for API Exceptions"""

    def __init__(self, code=None, reason=None, response=None):
        if response is not None:
            self.code = response.status_code
            try:
                result = response.json()
                if 'reason' in result:
                    self.reason = result['reason']
                elif 'error' in result:
                    self.reason = result['error']
                elif 'detail' in result:
                    self.reason = result['detail']
                else:
                    self.reason = 'unknown error.'
            except:
                pass
        else:
            self.code = code
            self.reason = reason

    def __str__(self):
        return '({0}). Reason: {1}'.format(self.code, self.reason)


class DaasClient(object):
    """Python client library for DaaS"""

    def __init__(self, url, username, password, project=None):
        """Encapsulates a DaaS client, handling top level user API calls having to do with authorization and
        server management.
        :param url: URL for DaaS server.
        :param username: Username used to connect to DaaS.
        :param password: Passcode used to connect to DaaS.
        :param project: Project to save assets, if it's not specified, the project `default` is used, and created
            automatically when it does not exists. An error is thrown when the specified project does not exist.
        """
        self.url = DaasClient.__del_last_slash(url)
        auth_info = DaasClient.authorize(url, username, password)
        self.token = auth_info['token']
        self.headers = {'Authorization': 'Bearer {}'.format(self.token)}
        self.url_prefix = '{url}/api/v1'.format(url=self.url)
        self.project = project
        if project:
            if not self.project_exists(project):
                raise ValueError(
                    'Project "{project}" not found'.format(project=project))

            project_info = self.get_project(project)
            self.route = project_info['route']
        else:
            # create a default project with default route
            current_user = self.get_current_user()
            self.uid = current_user['uid']
            self.project = 'default'
            self.route = 'uid' + str(self.uid)
            if not self.project_exists(self.project):
                if not self.create_project(name=self.project,
                                           route=self.route,
                                           description='A default project'):
                    raise ValueError(
                        'Failed to create the default project, try again later')

    @staticmethod
    def authorize(url, username, password):
        url = '{url}/api/v1/auth/validate'.format(url=DaasClient.__del_last_slash(url))
        response = requests.get(url,
                                auth=(username, password),
                                verify=False)
        if response.ok:
            result = response.json()
            if result['status'] == 'error':
                raise ValueError('Invalid username or password.')
            return result
        else:
            raise ApiException(response)

    def get_current_user(self):
        url = '{url_prefix}/user/currentuser'.format(url_prefix=self.url_prefix)
        response = requests.get(url, headers=self.headers, verify=False)
        if response.ok:
            return response.json()
        else:
            raise ApiException(response)

    def project_exists(self, name):
        url = '{url_prefix}/projects/{name}/check'.format(url_prefix=self.url_prefix, name=name)
        response = requests.get(url, headers=self.headers, verify=False)
        if response.ok:
            return response.json()['exist']
        else:
            return False

    def create_project(self, name, route, description=None):
        """Create a project in DaaS
        :param name: Project name, should be a valid directory name.
        :param route: Route is used as a project deployment identifier within the REST paths. It consists of lower case
            alphanumeric characters (a-z, and 0-9), with the - character allowed anywhere except the first or last
            character.
        :param description: Project description, optional
        :return: True if success, otherwise an exception raised.
        """
        url = '{url_prefix}/projects'.format(url_prefix=self.url_prefix)
        response = requests.post(url,
                                 headers=self.headers,
                                 verify=False,
                                 json={
                                     'name': name,
                                     'route': route,
                                     'description': description
                                 })
        if response.ok:
            return True
        else:
            raise ApiException(response)

    def get_project(self, name):
        url = '{url_prefix}/projects/{name}'.format(url_prefix=self.url_prefix, name=name)
        response = requests.get(url,
                                headers=self.headers,
                                verify=False)
        if response.ok:
            return response.json()
        else:
            raise ApiException(response)

    def set_project(self, name):
        if not self.project_exists(name):
            raise ValueError('Project "{name}" not found'.format(name=name))
        self.project = name
        project_info = self.get_project(name)
        self.route = project_info['route']

    def publish(self,
                model,
                name,
                mining_function=None,
                X_test=None,
                y_test=None,
                data_test=None,
                description=None,
                params=None):
        """Publish the model to DaaS
        :param model: The model object
        :param name: The model name identifies the model in the project.
            The model will be the first version if the specified model name not used, otherwise it could be a new
            version for such model name.
        :param mining_function: 'classification', 'regression', 'clustering'.
            Set the mining function of the model, could be inferred when not specified
        :param X_test: {array-like, sparse matrix}, shape (n_samples, n_features)
            Perform prediction on samples in X_test, predicted labels or estimated target values returned by the model.
        :param y_test: 1d array-like, or label indicator array / sparse matrix.
            Ground truth (correct) target values.
        :param data_test: Test dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
            Used by models of PySpark.
        :param description: Model description.
        :param params: An optional parameters to save.
        :return: dict
            model_name       The model published
            model_version    The model version published
            message          An optional message
        """
        metadata = get_model_metadata(model,
                                      mining_function=mining_function,
                                      X_test=X_test,
                                      y_test=y_test,
                                      data_test=data_test)

        if params is not None:
            metadata['params'] = params

        temp_path = os.path.join(tempfile.mkdtemp(), name)
        save_model(model, temp_path, metadata['serialization'])

        if os.path.isdir(temp_path):
            temp_path = shutil.make_archive(
                temp_path, 'zip', root_dir=temp_path)

        files = {'file': open(temp_path, 'rb')}
        url = '{url_prefix}/projects/{project}/models'.format(
            url_prefix=self.url_prefix,
            project=self.project)

        data = {
            'name': name,
            'type': metadata['type'],
            'description': description,
            'metadata': json.dumps(metadata)
        }

        response = requests.post(url,
                                 headers=self.headers,
                                 verify=False,
                                 files=files,
                                 data=data)
        if not response.ok:
            raise ApiException(response=response)

        result_json = response.json()
        result = {
            'model_name': name,
            'model_version': result_json['version']
        }

        if 'message' in result_json:
            result['message'] = result_json['message']

        return result

    @staticmethod
    def choose_runtime(version_info):
        result = 'daas-python37-faas'
        runtime = version_info['runtime']
        if runtime:
            runtime_to_test = runtime.lower().replace(' ', '-')
            if runtime_to_test.find('python2') != -1 or runtime_to_test.find('python-2') != -1:
                result = 'daas-python27-faas';
        return result

    @staticmethod
    def get_sample_value(typ):
        result = 'STRING_VALUE'
        if typ in ('double', 'float'):
            result = 'DOUBLE_VALUE'
        elif typ == 'int':
            result = 'INT_VALUE'
        return result

    @staticmethod
    def get_model_sample_values(model_info):
        result = {}
        features = model_info['features']
        predictors = model_info['version'].get('predictors')
        inputs = predictors or features
        for x in inputs:
            result[x['name']] = x['sample'] if 'sample' in x else DaasClient.get_sample_value(x['type'])
        return result

    def test(self, model_name, model_version='latest'):
        """Get model test information in DaaS
        :param model_name: The specified model to test
        :param model_version: The specified model version to test, default 'latest'
        :return: dict
            endpoint_url:    Endpoint url of model test to use.
            access_token:    Authorization access token to use in the next post request, e.g.
                Authorization: Bearer access_token
            payload:         JSON payload format used, it's an object, and all parameters in 'args':
                model_name:     required
                model_version:  optional, default 'latest'
                X:              JSON string format could be in either:
                    'split': dict like {columns -> [columns], data -> [values]}
                    'records': list like [{column -> value}, ... , {column -> value}]
        """
        # get model info
        url = '{url_prefix}/projects/{project}/models/{model}/versions/{version}'.format(
            url_prefix=self.url_prefix,
            project=self.project,
            model=model_name,
            version=model_version)
        response = requests.get(url, headers=self.headers, verify=False)
        if not response.ok:
            raise ApiException(response=response)
        model_info = response.json()

        # start the runtime
        runtime = self.choose_runtime(model_info['version'])
        url = '{url_prefix}/projects/{project}/runtimes/{runtime}/test-function-as-a-service'.format(
            url_prefix=self.url_prefix,
            project=self.project,
            runtime=runtime)
        response = requests.put(url,
                                headers=self.headers,
                                verify=False,
                                json={})
        if not response.ok or response.json()['status'] == 'error':
            raise ApiException(response=response)
        print('The runtime "{runtime}" is starting'.format(runtime=runtime))
        print('Waiting for it becomes available... \n')

        # get the runtime status
        # wait for max 90 * 2 = 180 sec
        url = '{url_prefix}/projects/{project}/runtimes/{runtime}/status'.format(
            url_prefix=self.url_prefix,
            project=self.project,
            runtime=runtime)
        running = False
        for x in range(90):
            response = requests.get(url, headers=self.headers, verify=False)
            if response.ok:
                status = response.json()['status']
                if status == 'Running':
                    running = True
                    break
                if status == 'Pending' or status == 'Starting':
                    time.sleep(2)
                else:
                    raise RuntimeError('Runtime "{runtime}" is in {status}, try again later'.format(runtime=runtime,
                                                                                                    status=status))
            else:
                raise ApiException(response=response)

        if not running:
            raise ValueError('Runtime can not be launched, try again later')

        return {
            'endpoint_url': '{host}/api/v1/test/{route}/{runtime}/test'.format(
                host=self.url,
                route=self.route,
                runtime=runtime),
            'access_token': self.token,
            'payload': {
                'args': {
                    'model_name': model_name,
                    'model_version': model_version,
                    'X': [self.get_model_sample_values(model_info)]
                }
            }}

    def deploy(self, model_name, deployment_name, model_version='latest', cpu=None, memory=None, replicas=1):
        """Deploy a model in DaaS
        :param model_name: The specified model to deploy
        :param deployment_name: An unique name identifies the model deployment.
        :param model_version: The specified Model version to deploy, default 'latest'
        :param cpu: float, how many cpu cores to assign this deployment runtime environment, default None
        :param memory: float, how many memory(GB) assign this deployment runtime environment, default None
        :param replicas: int, how many replicas for this deployment, default 1
        :return: dict
            access_token:   Authorization access token to use in the next post request, e.g.
                Authorization: Bearer access_token
            payload:        JSON payload format used, it's an object, and all parameters in 'args':
                X:  JSON string format could be in either:
                    'split': dict like {columns -> [columns], data -> [values]}
                    'records': list like [{column -> value}, ... , {column -> value}]
        """
        # get model info
        url = '{url_prefix}/projects/{project}/models/{model}/versions/{version}'.format(
            url_prefix=self.url_prefix,
            project=self.project,
            model=model_name,
            version=model_version)
        response = requests.get(url, headers=self.headers, verify=False)
        if not response.ok:
            raise ApiException(response=response)
        model_info = response.json()

        # deploy a service
        runtime = self.choose_runtime(model_info['version'])
        url = '{url_prefix}/projects/{project}/services'.format(
            url_prefix=self.url_prefix,
            project=self.project)
        response = requests.post(url,
                                 headers=self.headers,
                                 verify=False,
                                 json={
                                     'name': deployment_name,
                                     'type': 'Default scoring',
                                     'assetType': 'models',
                                     'assetName': model_name,
                                     'assetVersion': model_version,
                                     'scriptPath': 'scoring-scripts/default_scoring.py',
                                     'runtime': runtime,
                                     'environment': {
                                         'replicas': replicas,
                                         'cpu': cpu if cpu is not None and cpu > 0.0 else 0.0,
                                         'memory': memory if memory is not None and memory > 0.0 else 0.0
                                     }
                                 })
        if not response.ok or response.json()['status'] == 'error':
            raise ApiException(response=response)
        print('The deployment "{deployment}" created successfully'.format(deployment=deployment_name))
        print('Waiting for it becomes available... \n')

        # get pods to check their status
        # wait for max 90 * 2 = 180 sec
        url = '{url_prefix}/projects/{project}/services/{service}/pods'.format(
            url_prefix=self.url_prefix,
            project=self.project,
            service=deployment_name
        )
        for x in range(90):
            response = requests.get(url, headers=self.headers, verify=False)
            if response.ok:
                pods = response.json()
                if DaasClient.__is_pod_running(pods):
                    break
                if DaasClient.__all_pods_failed(pods):
                    raise RuntimeError('All pods failed, try again later')
                else:
                    time.sleep(2)

        # get the service deployment
        url = '{url_prefix}/projects/{project}/services/{service}'.format(
            url_prefix=self.url_prefix,
            project=self.project,
            service=deployment_name)
        response = requests.get(url, headers=self.headers, verify=False)
        if not response.ok:
            raise ApiException(response=response)
        deployment_info = response.json()

        return {
            'endpoint_url': '{host}/api/v1/svc/{route}/{name}/predict'.format(
                host=self.url,
                route=self.route,
                name=deployment_name
            ),
            'access_token': deployment_info['token'],
            'payload': {
                'args': {
                    'X': [self.get_model_sample_values(model_info)]
                }
            }}

    @staticmethod
    def __is_pod_running(pods):
        for x in pods:
            if x['status'] == 'Running':
                return True
        return False

    @staticmethod
    def __all_pods_failed(pods):
        for x in pods:
            if x['status'] in ('Running', 'Pending', 'Starting'):
                return False
        return True

    @staticmethod
    def __del_last_slash(url):
        return url[:-1] if url[-1] == '/' else url
