import cv2

class SkeletonAnimator:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame = None  # Zmienna do przechowywania klatki wideo

    def set_frame(self, frame):
        """Ustawia klatkę wideo, na której będzie rysowany szkielet."""
        self.frame = frame

    def draw_skeleton(self, keypoints):
        """Rysuje szkielet na podstawie kluczowych punktów."""
        if self.frame is None:
            raise ValueError("Frame is not set. Please set the frame using set_frame() before drawing the skeleton.")

        # Definicja połączeń między punktami szkieletu
        skeleton_connections = [
            (0, 1), (0, 2),  # Nose -> Eyes
            (1, 3), (2, 4),  # Eyes -> Ears
            (5, 6),  # Shoulders
            (5, 7), (6, 8),  # Shoulders -> Elbows
            (7, 9), (8, 10),  # Elbows -> Wrists
            (5, 11), (6, 12),  # Shoulders -> Hips
            (11, 13), (12, 14),  # Hips -> Knees
            (13, 15), (14, 16),  # Knees -> Ankles
        ]

        # Rysowanie punktów kluczowych
        for i in range(len(keypoints)):
            x, y, _ = keypoints[i]
            cv2.circle(self.frame, (int(x * self.frame_width), int(y * self.frame_height)), 5, (0, 255, 0), -1)

        # Rysowanie połączeń między punktami
        for connection in skeleton_connections:
            p1, p2 = connection
            x1, y1, _ = keypoints[p1]
            x2, y2, _ = keypoints[p2]

            # Sprawdzenie, czy oba punkty istnieją (mają wartość większą od 0)
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(self.frame, (int(x1 * self.frame_width), int(y1 * self.frame_height)),
                         (int(x2 * self.frame_width), int(y2 * self.frame_height)), (255, 0, 0), 2)

    def show(self):
        """Pokazuje końcowy obraz z rysowanym szkieletem."""
        if self.frame is not None:
            cv2.imshow("Skeleton Animation", self.frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
