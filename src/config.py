class config:
    def __init__(self) -> None:
        self.RANDOM_SEED = 42
        self.input_size = 768
        self.hidden_size_1 = 256
        self.hidden_size_2 = 256
        self.num_classes = 2
        self.batch_size = 256
        self.learning_rate = 5e-4
        self.num_epochs = 40
        self.NO_CLONE_PAIRS_TRAIN_NUM = 100000
        self.NO_CLONE_PAIRS_TEST_NUM = 10000

    def load_json(self, path):
        import json

        with open(path, "r") as f:
            data = json.load(f)

        self.RANDOM_SEED = data["RANDOM_SEED"]
        self.input_size = data["input_size"]
        self.hidden_size_1 = data["hidden_size_1"]
        self.hidden_size_2 = data["hidden_size_2"]
        self.num_classes = data["num_classes"]
        self.batch_size = data["batch_size"]
        self.learning_rate = data["learning_rate"]
        self.num_epochs = data["num_epochs"]
        self.NO_CLONE_PAIRS_TRAIN_NUM = data["NO_CLONE_PAIRS_TRAIN_NUM"]
        self.NO_CLONE_PAIRS_TEST_NUM = data["NO_CLONE_PAIRS_TEST_NUM"]
