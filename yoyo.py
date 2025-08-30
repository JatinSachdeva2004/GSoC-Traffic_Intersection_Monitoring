from ultralytics import YOLO

model = YOLO(r"D:\Downloads\Khatam2\khatam\qt_app_pyside\best4.pt")
model.export(format="openvino")