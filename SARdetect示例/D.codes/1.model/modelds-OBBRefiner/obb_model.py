"""Six-output EfficientNet-B0 OBB geometry model; state keys match deployed weights."""
from model_loader import load_model_module
Heads=load_model_module('model-c-s','hierarchy_model').Heads
class OBBRefiner(Heads):
    def __init__(self,tax):super().__init__(tax,"refiner")
