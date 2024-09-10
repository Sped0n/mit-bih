import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a vector from a numpy array."
    )
    parser.add_argument(
        "letter",
        type=str,
        help="Capital letter corresponding to the .npy file",
    )
    parser.add_argument("index", type=int, help="Index of the vector to visualize")

    args = parser.parse_args()

    # validate letter argument
    if not args.letter.isupper() or len(args.letter) != 1:
        print("Error: First argument must be a single capital letter.")
        return

    file_name = f"./{args.letter}.npy"

    try:
        data: NDArray[np.float32] = np.load(file_name)

        # check shape
        if len(data.shape) != 2 or data.shape[1] != 300:
            print(f"Error: {file_name} should have shape (N, 300)")
            return

        # check index boundary
        if args.index < 0 or args.index >= data.shape[0]:
            print(
                f"Error: Index out of bounds. Should be between 0 and {data.shape[0] - 1}"
            )
            return

        vector = data[args.index]

        plt.figure(figsize=(12, 6))
        plt.plot(vector)
        plt.title(f"Vector visualization for {file_name}, index {args.index}")
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()
    except FileNotFoundError:
        print(f"Error: File {file_name} not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
