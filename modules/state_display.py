import cv2

from matplotlib import pyplot as plt  # For plotting graphs and images


def display_state(state):
    """ Display current state according to player and agent"""

    """Scaled"""
    scaledObservation = cv2.resize(state, dsize=(30, 30), interpolation=cv2.INTER_LINEAR)

    """Gray and Scaled"""
    grayScaledObservation = cv2.cvtColor(scaledObservation, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(15, 12))

    plt.subplot(131)
    plt.imshow(state, cmap='gray')
    plt.title('Original State\n210 x 160 x 3 ndarray of pixels with colour range 0 - 184"')

    plt.subplot(132)
    plt.imshow(scaledObservation, cmap='gray')
    plt.title('Scaled State\n80 x 80 x 3 ndarray of pixels with colour range 0 - 184')

    plt.subplot(133)
    plt.imshow(grayScaledObservation, cmap='gray')
    plt.title('Gray Scaled State\n80 x 80 ndarray of pixels with gray range 0 - 163')

    plt.show()