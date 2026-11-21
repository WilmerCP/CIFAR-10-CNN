# entrenar.py
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim

from datos import get_loaders
from modelo import CNN

EPOCAS = 20
LR = 0.01
MOMENTUM = 0.9
BATCH = 64
device = "cpu"


def evaluar(model, loader, criterion):
    model.eval()
    loss_total, correctos, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            salida = model(x)
            loss = criterion(salida, y)
            loss_total += loss.item() * x.size(0)
            correctos += (salida.argmax(dim=1) == y).sum().item()
            n += x.size(0)
    return loss_total / n, correctos / n


def main():
    torch.manual_seed(42)
    train_loader, val_loader, _ = get_loaders(batch_size=BATCH)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    mejor_val_acc = 0.0
    inicio = time.time()

    for epoca in range(1, EPOCAS + 1):
        t0 = time.time()
        model.train()
        loss_total, correctos, n = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            salida = model(x)
            loss = criterion(salida, y)
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * x.size(0)
            correctos += (salida.argmax(dim=1) == y).sum().item()
            n += x.size(0)

        scheduler.step()
        train_loss, train_acc = loss_total / n, correctos / n
        val_loss, val_acc = evaluar(model, val_loader, criterion)

        historial["train_loss"].append(train_loss)
        historial["train_acc"].append(train_acc)
        historial["val_loss"].append(val_loss)
        historial["val_acc"].append(val_acc)

        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            torch.save(model.state_dict(), "modelo.pth")

        print(f"Época {epoca:02d}/{EPOCAS} | "
              f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} | "
              f"{time.time() - t0:.0f}s")

    tiempo_total = time.time() - inicio
    historial["tiempo_entrenamiento_s"] = tiempo_total
    historial["mejor_val_acc"] = mejor_val_acc
    with open("historial.json", "w") as f:
        json.dump(historial, f)

    print(f"\nEntrenamiento terminado en {tiempo_total / 60:.1f} min")
    print(f"Mejor accuracy de validación: {mejor_val_acc:.3f}")


if __name__ == "__main__":
    main()