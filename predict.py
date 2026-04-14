import torch
from PIL import Image, ImageOps
from torchvision import transforms
from model import CNNModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNModel().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()

    image = Image.open("digit.png").convert("L")

    # 視情況反相：如果你的圖片是白底黑字，MNIST通常比較像黑底白字
    image = ImageOps.invert(image)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()

    print("Prediction:", pred)

if __name__ == "__main__":
    main()