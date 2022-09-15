import data
import train
from pathlib import Path

if __name__ == "__main__":
    train.run_training(Path("."), "add", data.train, data.test)
