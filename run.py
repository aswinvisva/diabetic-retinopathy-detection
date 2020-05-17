from detector_model import Detector

if __name__ == '__main__':
    dct = Detector(should_load_model=True)
    # dct.train()
    results = dct.evaluate()
    dct.show_predictions()
    print(results)