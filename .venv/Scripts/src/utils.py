def normalize_keypoints(keypoints, width, height):
    """
    Normalizuje kluczowe punkty względem szerokości i wysokości obrazu.
    :param keypoints: Lista kluczowych punktów w formacie [x, y, confidence].
    :param width: Szerokość obrazu.
    :param height: Wysokość obrazu.
    :return: Lista znormalizowanych kluczowych punktów w formacie [x_norm, y_norm, confidence].
    """
    return [(x / width, y / height, confidence) for x, y, confidence in keypoints]
