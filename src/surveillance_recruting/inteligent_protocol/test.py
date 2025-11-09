import numpy as np

def main():
    map = np.zeros((10, 10, 2))
    print(f"Initial map: {map}")

    map[:, :,1] = map[:, :, 1] + 10

    print(f"Modified map: {map}")




if __name__ == "__main__":
    main()
