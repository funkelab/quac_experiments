import torch
from torchvision.utils import save_image


def main():
    image = torch.randn(1, 3, 224, 224)
    print(image.min(), image.max())
    save_image(image, "test.png")


if __name__ == "__main__":
    main()
