"""Сравнение val-метрик двух детекторов ценника на общем датасете NIK (43_15 val)."""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
# У коллеги: 2 полных кадра из 43_15 (удержанный по видео минивал).
DATA_MINIVAL = ROOT / "NIK/lenta_tech_ml/experiments/prepared/price_tag_yolo_val_43_15/dataset_local.yaml"
# Плитки 768×768, val по источникам 25_2-10 и 49_5 (не участвовали в их «oldtrain»).
DATA_TILES_VAL = (
    ROOT
    / "NIK/lenta_tech_ml/experiments/price_tag_yolo_detect_tiles768_fulltags_oldtrain_newval_v1"
    / "dataset_local.yaml"
)
W_OURS = ROOT / "runs/detect/runs/detect/price_tag_merged/weights/best.pt"
W_NIK = (
    ROOT
    / "NIK/lenta_tech_ml/experiments/price_tag_detector_runs"
    / "detect_yolo11n_tiles768_fulltags_oldtrain_newval_e12_i640/weights/best.pt"
)


def run_val(data_yaml: Path, weights: Path, imgsz: int) -> dict[str, float]:
    m = YOLO(str(weights))
    r = m.val(data=str(data_yaml), imgsz=imgsz, batch=4, device="cpu", verbose=False, plots=False)
    box = r.box
    return {
        "P": float(box.mp),
        "R": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }


def compare_on_dataset(label: str, data_yaml: Path, imgsz_list: tuple[int, ...]) -> None:
    print(f"\n{'=' * 60}\n{label}\ndata: {data_yaml.resolve()}\n{'=' * 60}")
    if not data_yaml.is_file():
        print(f"MISSING yaml: {data_yaml}")
        return
    for imgsz in imgsz_list:
        print(f"\n--- imgsz={imgsz} ---")
        for tag, w in [
            ("ours_merged (YOLOv8n)", W_OURS),
            ("NIK colleague (YOLO11n detect)", W_NIK),
        ]:
            if not w.is_file():
                print(f"  {tag}: MISSING {w}")
                continue
            try:
                met = run_val(data_yaml, w, imgsz)
                print(
                    f"  {tag}: P={met['P']:.4f} R={met['R']:.4f} "
                    f"mAP50={met['mAP50']:.4f} mAP50-95={met['mAP50-95']:.4f}"
                )
            except Exception as e:
                print(f"  {tag}: ERROR {e}")


def main() -> None:
    compare_on_dataset(
        "Минивал: полные кадры 43_15 (2 изображения; метрики шумные)",
        DATA_MINIVAL,
        (640, 1280),
    )
    compare_on_dataset(
        "Вал коллеги: плитки 768, источники 25_2-10 и 49_5 (YOLO11n обучался на таких плитках)",
        DATA_TILES_VAL,
        (640,),
    )


if __name__ == "__main__":
    main()
