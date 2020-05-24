from imageai.Prediction.Custom import ModelTraining
model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("code")
model_trainer.trainModel(num_objects=2, num_experiments=5, batch_size=2, show_network_summary=True)

