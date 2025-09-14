from bases.base import TrainModels
from retail_task.src.data_preprocessing import RetailModel


if __name__ == "__main__":
    l = RetailModel()
    l.preprocess()
    t = TrainModels(l, "avg_purchase_value")
    t.train_and_save(l.folder_name, "model1")

