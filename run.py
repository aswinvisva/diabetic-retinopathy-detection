from diagnosis_pipeline.detector_model import Detector

def k_fold_cross_validation(k=15):
    k_fold_cross_validation_list = []

    for i in range(k):
        dct = Detector()
        dct.train()
        results = dct.evaluate()
        accuracy = results[1]
        k_fold_cross_validation_list.append(accuracy)

    average = sum(k_fold_cross_validation_list) / len(k_fold_cross_validation_list)

    return average

def show_prediction():
    dct = Detector(should_load_model=True)
    dct.train()
    dct.show_predictions()

if __name__ == '__main__':
    average = k_fold_cross_validation()
    print(average)