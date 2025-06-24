from pathlib import Path
import os
import cv2

ROOT_DIR = Path(__file__).resolve().parent

CASCADE_PATH = ROOT_DIR / 'model' / 'haarcascade_frontalface_default.xml'
FACE_CASCADE = cv2.CascadeClassifier(str(CASCADE_PATH))

def process_directory(src_root: Path, dest_root: Path) -> None:
    for root, _, files in os.walk(src_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = Path(root) / file
                rel_path = src_path.relative_to(src_root)
                dest_path = dest_root / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    img = cv2.imread(str(src_path))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces_list = FACE_CASCADE.detectMultiScale(gray, 1.5, 5)
                    for idx, (x, y, w, h) in enumerate(faces_list):
                        roi = img[y:y + h, x:x + w]
                        if len(faces_list) > 1:
                            stem = dest_path.stem
                            ext = dest_path.suffix
                            face_filename = dest_path.with_name(f"{stem}_face_{idx}{ext}")
                        else:
                            face_filename = dest_path
                        cv2.imwrite(str(face_filename), roi)
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")


# Example usage
def main() -> None:
    female_src = Path('./data/images/train/female/known')
    female_dst = Path('./data/images/crop/female/known')
    male_src = Path('./data/images/train/male/known')
    male_dst = Path('./data/images/crop/male/known')
    process_directory(female_src, female_dst)
    process_directory(male_src, male_dst)

if __name__ == "__main__":
    main()