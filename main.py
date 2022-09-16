import data
import train
from pathlib import Path

if __name__ == "__main__":
    train.run_training(Path("."), "div", data.div_train, data.div_test)
