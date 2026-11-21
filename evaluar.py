# evaluar.py
import json
import time
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

from datos import get_loaders, CLASES
from modelo import CNN

device = "cpu"


def main():
    _, _, test_loader = get_loaders(batch_size=64)

    model = CNN().to(device)
    model.load_state_dict(torch.load("modelo.pth", map_location=device))
    model.eval()

    # Predicciones sobre las 10.000 imágenes de test
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            salida = model(x.to(device))
            y_pred.extend(salida.argmax(dim=1).cpu().tolist())
            y_true.extend(y.tolist())

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (macro)")
    print(f"Recall:    {rec:.4f} (macro)")
    print(f"F1-score:  {f1:.4f} (macro)\n")
    print(classification_report(y_true, y_pred, target_names=CLASES, digits=3))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=150)
    plt.close()

    # Curvas de entrenamiento (desde historial.json)
    with open("historial.json") as f:
        h = json.load(f)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(h["train_loss"], label="train")
    a1.plot(h["val_loss"], label="validación")
    a1.set_title("Loss"); a1.set_xlabel("Época"); a1.legend()
    a2.plot(h["train_acc"], label="train")
    a2.plot(h["val_acc"], label="validación")
    a2.set_title("Accuracy"); a2.set_xlabel("Época"); a2.legend()
    plt.tight_layout()
    plt.savefig("curvas.png", dpi=150)
    plt.close()

    # Tiempo de inferencia: una imagen a la vez, promedio de 200
    imagenes = [test_loader.dataset[i][0].unsqueeze(0) for i in range(200)]
    with torch.no_grad():
        for img in imagenes[:10]:          # calentamiento
            model(img)
        t0 = time.perf_counter()
        for img in imagenes:
            model(img)
        ms = (time.perf_counter() - t0) / len(imagenes) * 1000
    print(f"\nTiempo de inferencia: {ms:.2f} ms por imagen")

    with open("metricas.json", "w") as f:
        json.dump({"accuracy": acc, "precision_macro": prec, "recall_macro": rec,
                   "f1_macro": f1, "inferencia_ms": ms}, f, indent=2)


if __name__ == "__main__":
    main()