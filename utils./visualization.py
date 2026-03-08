import matplotlib.pyplot as plt
import numpy as np


def show_stress_map(stress_map):

    plt.figure(figsize=(6,6))

    plt.imshow(stress_map, cmap="jet")

    plt.title("Predicted Stress Map")

    plt.colorbar()

    plt.axis("off")

    plt.show()


def compare_stress_maps(predicted, ground_truth):

    fig, axs = plt.subplots(1,2,figsize=(10,5))

    axs[0].imshow(predicted, cmap="jet")
    axs[0].set_title("Predicted Stress")

    axs[1].imshow(ground_truth, cmap="jet")
    axs[1].set_title("Ground Truth Stress")

    for ax in axs:
        ax.axis("off")

    plt.show()


def plot_force_prediction(pred_force, true_force):

    pred_force = np.array(pred_force)
    true_force = np.array(true_force)

    plt.figure()

    plt.plot(true_force[:,0], label="True Fx")
    plt.plot(pred_force[:,0], label="Predicted Fx")

    plt.xlabel("Samples")
    plt.ylabel("Force")

    plt.legend()

    plt.title("Force Prediction vs Ground Truth")

    plt.show()
