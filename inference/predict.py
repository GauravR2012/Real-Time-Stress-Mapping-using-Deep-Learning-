import torch
import cv2
import argparse

from models.fre_network import FREModel


def preprocess(image_path):

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    image = image.unsqueeze(0)

    return image


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="results/model_weights.pth")

    args = parser.parse_args()

    model = FREModel()

    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    model.eval()

    image = preprocess(args.image)

    with torch.no_grad():

        stress, force = model(image)

    print("Predicted Force:", force.squeeze().numpy())

    stress_map = stress.squeeze().numpy()

    cv2.imwrite("results/predicted_stress_map.png", stress_map * 255)


if __name__ == "__main__":
    main()
