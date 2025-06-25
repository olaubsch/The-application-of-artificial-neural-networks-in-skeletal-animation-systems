from ultralytics import YOLO

class YOLOHandler:
    def __init__(self, model_path):
        """
        Inicjalizuje model YOLO.
        :param model_path: Ścieżka do pliku modelu YOLO.
        """
        self.model = YOLO(model_path)

    def predict(self, frame):
        """
        Przeprowadza predykcję na pojedynczej klatce.
        :param frame: Obraz w formacie NumPy (klatka wideo).
        :return: Wyniki predykcji YOLO.
        """
        results = self.model.predict(frame, conf=0.5)
        return results
