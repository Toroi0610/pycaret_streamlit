from app import Pycaret_CLI


pcl = Pycaret_CLI("setting_automl.csv")

train, test = pcl.load_data()
exp = pcl.setup_automl_env(train)
best_model = pcl.training_model()
pred_result = pcl.prediction(best_model, test)

