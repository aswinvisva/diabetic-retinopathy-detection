from diagnosis_pipeline.detector_model import Detector

def k_fold_cross_validation(k=15):
    k_fold_cross_validation_list = []

    for i in range(k):
        dct = Detector(train_split=0.99)
        dct.train()
        results = dct.evaluate()
        accuracy = results[1]
        k_fold_cross_validation_list.append(accuracy)

    average = sum(k_fold_cross_validation_list) / len(k_fold_cross_validation_list)

    return average

def train():
    dct = Detector(train_split=0.99, use_transfer_learning_ensemble=True)
    dct.train()
    results = dct.evaluate()
    print(results)
    dct.show_predictions()

def show_prediction():
    dct = Detector(should_load_model=True)
    dct.show_predictions()

def evaluate():
    dct = Detector(should_load_model=True)
    print(dct.evaluate())

if __name__ == '__main__':
    train()