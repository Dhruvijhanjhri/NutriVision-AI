from src.model_builder import build_mobilenet

model, base_model = build_mobilenet(num_classes=30)

model.summary()