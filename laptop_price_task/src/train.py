from bases.base import TrainModels
from laptop_price_task.src.data_preprocessing import LaptopModel


if __name__ == "__main__":
    l = LaptopModel()
    l.preprocess()
    t = TrainModels(l, "Price")
    t.train_and_save(l.folder_name, "final_model")
