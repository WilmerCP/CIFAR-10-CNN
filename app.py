# app.py
import torch
import gradio as gr
from PIL import Image, ImageOps

from datos import CLASES, eval_tf
from modelo import CNN

device = "cpu"
model = CNN().to(device)
model.load_state_dict(torch.load("modelo.pth", map_location=device))
model.eval()


def preparar(img):
    img = ImageOps.exif_transpose(img).convert("RGB")           # corrige rotación de fotos de teléfono
    return ImageOps.fit(img, (32, 32), Image.Resampling.LANCZOS)  # recorte cuadrado central + 32x32


def predecir(img):
    if img is None:
        return None, None
    pequena = preparar(img)
    x = eval_tf(pequena).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    resultado = {CLASES[i]: float(p) for i, p in enumerate(probs)}
    vista = pequena.resize((256, 256), Image.Resampling.NEAREST)  # lo que ve la red, ampliado
    return resultado, vista


demo = gr.Interface(
    fn=predecir,
    inputs=gr.Image(type="pil", label="Sube una foto"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predicción"),
        gr.Image(label="Lo que ve la red (32x32)"),
    ],
    title="Clasificador CIFAR-10 (CNN propia)",
    description="Clases: avión, auto, pájaro, gato, ciervo, perro, rana, caballo, barco, camión.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()