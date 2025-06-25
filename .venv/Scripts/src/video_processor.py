import cv2

class VideoProcessor:
    def __init__(self, video_path):
        """
        Inicjalizuje obiekt do przetwarzania wideo.
        :param video_path: Ścieżka do pliku wideo.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    def read_frame(self):
        """
        Czyta jedną klatkę z wideo.
        :return: Klatka wideo w formacie NumPy lub None, jeśli wideo się skończy.
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """
        Zwalnia zasoby wideo.
        """
        self.cap.release()
        cv2.destroyAllWindows()
