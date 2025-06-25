import cv2
import numpy as np
import time
from src.yolo_handler import YOLOHandler
from src.utils import normalize_keypoints
from src.skeleton import SkeletonAnimator
from src.video_processor import VideoProcessor
import fbx


class Main:
    def __init__(self, model_path, video_path, output_path, skeleton_output_path, fbx_output_path):
        """
        Inicjalizuje główne komponenty aplikacji.
        :param model_path: Ścieżka do modelu YOLO.
        :param video_path: Ścieżka do pliku wideo.
        :param output_path: Ścieżka do zapisanego wideo z narysowanym szkieletem (tło + szkielet).
        :param skeleton_output_path: Ścieżka do zapisanego wideo z samym szkieletem na czarnym tle.
        :param fbx_output_path: Ścieżka do zapisanego pliku FBX.
        """
        self.yolo_handler = YOLOHandler(model_path)
        self.video_processor = VideoProcessor(video_path)

        self.fbx_output_path = fbx_output_path

        # Otwórz wideo, aby uzyskać szerokość i wysokość klatki
        self.video = cv2.VideoCapture(video_path)

        if not self.video.isOpened():
            print("Błąd: Nie udało się otworzyć pliku wideo.")
            return

        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.skeleton_animator = SkeletonAnimator(self.frame_width, self.frame_height)

        # Inicjalizacja zapisu wideo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, 30.0, (self.frame_width, self.frame_height))
        self.skeleton_out = cv2.VideoWriter(skeleton_output_path, fourcc, 30.0, (self.frame_width, self.frame_height))

        # Tworzenie menedżera FBX
        self.fbx_manager = fbx.FbxManager.Create()
        self.scene = fbx.FbxScene.Create(self.fbx_manager, "MyScene")

    def export_to_fbx(self, all_keypoints, scale_factor=10.0):
        """
        Eksportuje kluczowe punkty jako animowany szkielet do pliku FBX.
        :param all_keypoints: Lista klatek, gdzie każda zawiera kluczowe punkty.
        :param scale_factor: Współczynnik skalowania pozycji kości.
        """
        anim_stack = fbx.FbxAnimStack.Create(self.scene, "AnimationStack")
        anim_layer = fbx.FbxAnimLayer.Create(self.scene, "BaseLayer")
        anim_stack.AddMember(anim_layer)

        # Tworzenie korzenia szkieletu
        skeleton_root_node = fbx.FbxNode.Create(self.scene, "Root")
        skeleton_root = fbx.FbxSkeleton.Create(self.scene, "Root_Skeleton")
        skeleton_root_node.SetNodeAttribute(skeleton_root)
        self.scene.GetRootNode().AddChild(skeleton_root_node)

        # Dodawanie kości w hierarchii
        bone_nodes = []
        for i in range(len(all_keypoints[0])):  # Dla każdego kluczowego punktu
            bone = fbx.FbxSkeleton.Create(self.scene, f"Bone_{i}")

            bone_node = fbx.FbxNode.Create(self.scene, f"Bone_{i}")
            bone_node.SetNodeAttribute(bone)

            # Jeśli to pierwsza kość, dodaj do korzenia, inaczej połącz z poprzednią kością
            if i == 0:
                skeleton_root_node.AddChild(bone_node)
            else:
                bone_nodes[-1].AddChild(bone_node)

            bone_nodes.append(bone_node)

        # Animacja pozycji kości
        for frame_idx, keypoints in enumerate(all_keypoints):
            for i, (x, y, _) in enumerate(keypoints):
                bone_node = bone_nodes[i]

                time_val = fbx.FbxTime()
                time_val.SetSecondDouble(frame_idx / 30.0)  # Przyjmujemy 30 FPS

                # Skalowanie współrzędnych
                scaled_x = x * scale_factor
                scaled_y = y * scale_factor

                # Pozycje względne względem rodzica
                if i > 0:
                    rel_x = scaled_x - (keypoints[i - 1][0] * scale_factor)
                    rel_y = scaled_y - (keypoints[i - 1][1] * scale_factor)
                else:
                    rel_x = scaled_x
                    rel_y = scaled_y

                for axis, value in zip("XYZ", (rel_x, -rel_y, 0)):
                    anim_curve = bone_node.LclTranslation.GetCurve(anim_layer, axis, True)
                    anim_curve.KeyModifyBegin()
                    key_index = anim_curve.KeyAdd(time_val)[0]
                    anim_curve.KeySetValue(key_index, value)
                    anim_curve.KeyModifyEnd()

        # Eksport sceny do pliku FBX
        exporter = fbx.FbxExporter.Create(self.fbx_manager, "")
        exporter.Initialize(self.fbx_output_path, -1, self.fbx_manager.GetIOSettings())
        exporter.Export(self.scene)
        exporter.Destroy()

        print(f"Zapisano szkielet do pliku FBX: {self.fbx_output_path}")

    def run(self):
        """
        Główna pętla aplikacji z logowaniem FPS.
        """
        all_keypoints = []
        start_time = time.time()
        frame_count = 0

        while True:
            frame = self.video_processor.read_frame()
            if frame is None:
                break

            frame_count += 1

            results = self.yolo_handler.predict(frame)

            for result in results:
                if hasattr(result, "keypoints") and result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()[0]
                    normalized_keypoints = normalize_keypoints(keypoints, frame.shape[1], frame.shape[0])
                    all_keypoints.append(normalized_keypoints)

                    self.skeleton_animator.set_frame(frame)
                    self.skeleton_animator.draw_skeleton(normalized_keypoints)

                    self.out.write(self.skeleton_animator.frame)

                    black_frame = 255 * np.ones_like(frame)
                    self.skeleton_animator.set_frame(black_frame)
                    self.skeleton_animator.draw_skeleton(normalized_keypoints)
                    self.skeleton_out.write(self.skeleton_animator.frame)

            cv2.imshow("YOLO Detection with Skeleton", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end_time = time.time()
        fps = frame_count / (end_time - start_time)

        print(f"Liczba przetworzonych klatek: {frame_count}")
        print(f"Całkowity czas przetwarzania: {end_time - start_time:.2f} sekund")
        print(f"Średnia liczba klatek na sekundę (FPS): {fps:.2f}")

        self.export_to_fbx(all_keypoints, scale_factor=20000.0)

        self.video_processor.release()
        self.out.release()
        self.skeleton_out.release()


if __name__ == "__main__":
    MODEL_PATH = "models/yolo11x-pose.pt"
    VIDEO_PATH = "data/videos/example3.mp4"
    OUTPUT_PATH = "data/videos/output_video_with_skeleton.mp4"
    SKELETON_OUTPUT_PATH = "data/videos/output_video_skeleton_only.mp4"
    FBX_OUTPUT_PATH = "data/exported_skeleton.fbx"

    main_app = Main(MODEL_PATH, VIDEO_PATH, OUTPUT_PATH, SKELETON_OUTPUT_PATH, FBX_OUTPUT_PATH)
    main_app.run()
