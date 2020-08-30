import os
from pathlib import Path
from shutil import copy2
import pandas as pd


setting = pd.read_csv("setting_automl.csv", index_col=0)
mode = setting.loc["mode", "property0"]
# Importing pycaret
if mode == "classification":
    from pycaret.classification import *
elif mode  == "regression":
    from pycaret.regression import *


class Pycaret_CLI:

    def __init__(self, path_setting_file):

        setting = pd.read_csv(path_setting_file, index_col=0)
        self.mode = setting.loc["mode", "property0"]
        self.path_train_file = setting.loc["path_training_file", "property0"]
        self.path_test_file = setting.loc["path_test_file", "property0"]
        # self.preprocessing = dict(setting.loc["preprocessing", "property0"])
        self.target = setting.loc["target", "property0"]
        self.module = setting.loc["module", "property0"]
        self.model_list = setting.loc["models"].dropna().values.tolist()
        self.metric = setting.loc["metric", "property0"]
        self.exp_name = setting.loc["exp_name", "property0"]
        self.ignore_features = setting.loc["ignore_features"].dropna().values

        exp_dir = Path(self.exp_name).absolute()
        exp_dir.mkdir(exist_ok=True)
        model_dir = (exp_dir / "model").absolute()
        model_dir.mkdir(exist_ok=True)
        data_dir = (exp_dir / "data").absolute()
        data_dir.mkdir(exist_ok=True)
        setting_dir = (exp_dir / "setting").absolute()
        setting_dir.mkdir(exist_ok=True)
        predict_dir = (exp_dir / "predcit").absolute()
        predict_dir.mkdir(exist_ok=True)

        copy2(path_setting_file, setting_dir)
        copy2(self.path_train_file, data_dir)
        copy2(self.path_test_file, data_dir)

        self.exp_dir = exp_dir
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.setting_dir = setting_dir
        self.predict_dir = predict_dir


    def load_data(self):
        """[summary]
        Load training, test data. Their path is written in setting file.

        Returns:
            [type]: [description]
        """
        train = pd.read_csv(self.path_train_file)
        test = pd.read_csv(self.path_test_file)

        return train, test


    def setup_automl_env(self, train):

        print("SETUP EXPERIMENTS")
        os.chdir(self.exp_dir)
        exp_0 = setup(data=train, target=self.target,
                      html=False, silent=True,
                      ignore_features=self.ignore_features,
                      log_experiment=True,
                      log_plots=True, log_profile=True,
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

        return best_model


    def prediction(self, model, test):
        result = predict_model(model, test)
        result.to_csv((self.predict_dir / "result.csv").absolute(), index=False)
        return result
