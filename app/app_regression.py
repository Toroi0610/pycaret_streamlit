import os
from pathlib import Path
from shutil import copy2
import pandas as pd
import mlflow


mode = "regression"
# Importing pycaret
if mode == "classification":
    from pycaret.classification import *
elif mode  == "regression":
    from pycaret.regression import *


class Pycaret_CLI:

    def __init__(self,
                 df_train,
                 df_test,
                 model_list,
                 target,
                 metric,
                 module="compare",
                 exp_name="experiment_0",
                 ignore_features=None
                ):

        # self.mode = mode
        # self.path_train_file = setting.loc["path_training_file", "property0"]
        # self.path_test_file = setting.loc["path_test_file", "property0"]
        # # self.preprocessing = dict(setting.loc["preprocessing", "property0"])
        # self.target = setting.loc["target", "property0"]
        # self.module = setting.loc["module", "property0"]
        # # self.model_list = setting.loc["models"].dropna().values.tolist()
        # self.model_list = model_list
        # self.metric = setting.loc["metric", "property0"]
        # self.exp_name = setting.loc["exp_name", "property0"]
        # self.ignore_features = setting.loc["ignore_features"].dropna().values

        self.mode = mode
        self.target = target
        self.module = module
        self.df_train = df_train
        self.df_test = df_test
        self.model_list = model_list
        self.metric = metric
        self.exp_name = exp_name
        self.ignore_features = ignore_features

        exp_dir = Path(self.exp_name).absolute()
        exp_dir.mkdir(exist_ok=True)
        model_dir = (exp_dir / "model").absolute()
        model_dir.mkdir(exist_ok=True)
        data_dir = (exp_dir / "data").absolute()
        data_dir.mkdir(exist_ok=True)
        # setting_dir = (exp_dir / "setting").absolute()
        # setting_dir.mkdir(exist_ok=True)
        result_dir = (exp_dir / "result").absolute()
        result_dir.mkdir(exist_ok=True)
        # model_image_dir = (exp_dir / "model" / "image").absolute()
        # model_image_dir.mkdir(exist_ok=True)

        # configure mlflow tracking
        # database
        # uri = f'sqlite:///{exp_dir.as_posix()}/experiment.db'
        uri = f'file:///{exp_dir.as_posix()}/mlruns'
        mlflow.set_tracking_uri(uri)

        # copy2(path_setting_file, setting_dir)
        # copy2(self.path_train_file, data_dir)
        # copy2(self.path_test_file, data_dir)
        df_train.to_csv(data_dir/"train.csv", index=False)
        df_test.to_csv(data_dir/"test.csv", index=False)

        self.exp_dir = exp_dir
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.mlflow_server = uri

        # artifact
        self.artifact_uri = f"{exp_dir.as_posix()}/mlruns"


    def load_data(self):
        """[summary]
        Load training, test data. Their path is written in setting file.

        Returns:
            [type]: [description]
        """
        train = pd.read_csv(self.path_train_file)
        test = pd.read_csv(self.path_test_file)

        return train, test


    def setup_automl_env(self, df_train=None):

        if df_train is None:
            df_train = self.df_train

        print("SETUP EXPERIMENTS")

        os.chdir(self.exp_dir)
        exp_0 = setup(data=df_train, target=self.target,
                      html=False, silent=True,
                      ignore_features=self.ignore_features,
                      log_experiment=True,
                      log_plots=True, log_profile=True,
                      log_data=True,
                      profile=True,
                      use_gpu=True,
                      experiment_name=self.exp_name)

        print("Finished !!")

        return exp_0


    def training_model(self):
        if self.module == "compare":
            if self.model_list == ["all"]:
                print("Compare models and get a best model")
                best_model = compare_models(sort=self.metric)
            else:
                best_model = compare_models(include=self.model_list, sort=self.metric)
        elif self.module == "tune":
            if self.model_list == ["all"]:
                print("Tune model")
                best_model = compare_models(sort=self.metric)
            else:
                best_model = tune_model(best_model, sort=self.metric)

        save_model(best_model, str((self.model_dir / "best_model").absolute()))

        self.best_model = best_model

        return best_model


    def prediction(self, model=None, df_test=None):
        if model is None:
            model = self.best_model
        if df_test is None:
            df_test = self.df_test
        predict = predict_model(model, df_test)
        predict.to_csv((self.result_dir / "predict.csv").absolute(), index=False)

        return predict


    def model_image(self, model):
        pass
