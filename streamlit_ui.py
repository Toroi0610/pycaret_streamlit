import os
import glob
import pandas as pd
import base64
import zipfile
from subprocess import call, Popen
from socket import gethostname, gethostbyname
from shutil import make_archive, rmtree
from pathlib import Path

import streamlit as st

pycaret_example_dir = Path(".").absolute()

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache
def delete_exp_env(path=pycaret_example_dir,
                   ignore_dir=["__pycache__",".git", "env"]):
    # from app import Pycaret_CLI
    try:
        remove_glob("*.zip")
    except:
        pass
    try:
        remove_glob("*.log")
    except:
        pass

    dir_list = [d for d in os.listdir(path=pycaret_example_dir) \
                if (os.path.isdir(d) and not d in ignore_dir)]
    for d in dir_list:
        rmtree(d, ignore_errors=True)

@st.cache
def remove_glob(pathname, recursive=True):
    for p in glob.glob(pathname, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)

@st.cache
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download result {file_label}</a>'
    return href


st.sidebar.image("logo.png", width=200)
st.sidebar.markdown(
    "# Auto-ML-GUI using pycaret by 3323 Y.Ogasawara"
)

exp_name = st.sidebar.text_input(
    "Input experiment name",
    "exp_0"
)

mode = st.sidebar.selectbox(
    "Input Analysis mode",
    ["classification", "regression"]
)

if st.sidebar.button("Initialize"):
    delete_exp_env()



def main():

    target = False
    ignore_features = False

    uploaded_training_file = None
    uploaded_test_file = None
    # uploaded_setting_file = None

    path_setting_file = "setting.csv"

    target = st.text_input("target")

    # uploaded_setting_file = st.file_uploader("Experiment setting file [.csv]",
    #                                          type=["csv"],
    #                                          encoding='auto',
    #                                          key=None)

    if mode == "regression":
        model_list = st.multiselect(
            'What are your favorite models',
            ["lr", "lasso", "ridge", "en", "lar", "llar", "omp", "br", "ard", "par", "ransac",
            "tr", "huber", "kr", "svm", "knn", "dt", "rf", "et", "ada", "gbr", "mlp", "xgboost",
            "lightgbm", "catboost"],
            ['lightgbm']
            )
        metric = st.selectbox(
            'Metric',
            ["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE", "TT (Sec)"]
            )
    elif mode == "classification":
        model_list = st.multiselect(
            'What are your favorite models',
            ["lr", "knn", "nb", "dt", "svm", "rbfsvm", "gpc", "mlp", "ridge", "rf",
            "qda", "ada", "gbc", "lda", "et", "xgboost", "lightgbm", "catboost"],
            ['lightgbm']
            )
        metric = st.selectbox(
            'Metric',
            ["Accuracy", "AUC","Recall","Precision","F1","Kappa","MCC","TT (Sec)"]
            )

    uploaded_training_file = st.file_uploader("Training data [.csv]",
                                              type=["csv"],
                                              encoding='auto',
                                              key=None)

    uploaded_test_file = st.file_uploader("Test data [.csv]",
                                          type=["csv"],
                                          encoding='auto',
                                          key=None)

    # print("Training_file: ", uploaded_training_file)
    if st.button("Start!!"):
        if uploaded_training_file is not None and uploaded_test_file is not None:
            work_dir = os.getcwd()
            base_setting = pd.read_csv("base_setting.csv", index_col=0)
            train = pd.read_csv(uploaded_training_file)
            test = pd.read_csv(uploaded_test_file)
            setting = base_setting.copy()
            setting.loc["exp_name", "property0"] = exp_name
            setting.loc["target", "property0"] = target
            setting.loc["models", "property0"] = "_".join(model_list)
            setting.loc["metric", "property0"] = metric
            setting.loc["mode", "property0"] = mode

            st.dataframe(setting)
            train.to_csv(setting.loc["path_training_file", "property0"], index=False)
            test.to_csv(setting.loc["path_test_file", "property0"], index=False)

            setting.to_csv(path_setting_file)

            with st.spinner("Training..."):
                print(mode)
                if mode == "regression":
                    from app_reg import Pycaret_CLI
                elif mode == "classification":
                    from app_cls import Pycaret_CLI

                pcl = Pycaret_CLI(path_setting_file, model_list)
                exp = pcl.setup_automl_env(train)
                best_model = pcl.training_model()
                pred_result = pcl.prediction(best_model, test)
                pcl.model_image(best_model)
                os.chdir(work_dir)
                st.success("Done!")

                make_archive(pcl.exp_name, 'zip', root_dir=pcl.exp_dir)
                st.markdown(get_binary_file_downloader_html(f'{pcl.exp_name}.zip', 'ZIP'), unsafe_allow_html=True)

                hostname = gethostname()
                href = f'<a href="http://{hostname}:5000" \
                        target="_blank" rel="noopener noreferrer">\
                        Move to mlflow server</a>'

                st.markdown(href, unsafe_allow_html=True)

                proc = Popen(['mlflow', 'ui', "-p", "5000", "-h", "0.0.0.0",\
                              "--backend-store-uri", pcl.mlflow_server],
                              # "--default-artifact-root", pcl.artifact_uri],
                            )

                if st.button("Stop MLflow"):
                    proc.kill()

                # delete_exp_env()
                # uploaded_training_file = None
                # uploaded_test_file = None
                # uploaded_setting_file = None


if __name__ == "__main__":
    main()
