import gdown
import argparse

# List of model URLs
model_url = [
    "https://drive.google.com/file/d/1aWVGwu497_k5U0iqUZCrSDiiYPb4NQwk/view?usp=drive_link",
    "https://drive.google.com/file/d/1v0YJhLZoSBBLHN0ez0IqyIUpye-qU57u/view?usp=drive_link",
    "https://drive.google.com/file/d/1p3wEnYygl48EZrqrBAdUyIz9CYmNBtT3/view?usp=drive_link",
    "https://drive.google.com/file/d/1IpPfmfxGzO8Q9phTiyuXGaq4k5AT_8tJ/view?usp=drive_link",
    "https://drive.google.com/file/d/1xHwUyEmeVnqZ71E48sVi8kOZGNjvKwA5/view?usp=drive_link",
    "https://drive.google.com/file/d/1wcakAXW32HJEcQ8w5buQ0R3qF4fFyyOB/view?usp=drive_link",
    "https://drive.google.com/file/d/1kbZNt0JXmuF1wMfM66MziKm6DyQgoaEK/view?usp=drive_link",
]


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Download a model file from a list of URLs using an index."
    )

    # Add the index argument
    parser.add_argument(
        "index", type=int, help="Index of the URL to download (0-based index)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate the index
    if args.index < 0 or args.index >= len(model_url):
        print(
            f"Error: Index {args.index} is out of range. Please provide an index between 0 and {len(model_url) - 1}."
        )
        return 1  # Return a non-zero exit status to indicate failure

    # Download the file using the specified index
    try:
        gdown.download(
            model_url[args.index],
            str(f"model_{args.index}.pt"),
            quiet=False,
            fuzzy=True,
        )
        print(f"Downloaded model from URL {args.index} successfully.")
        return 0  # Return a zero exit status to indicate success
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")
        return 1  # Return a non-zero exit status to indicate failure


if __name__ == "__main__":
    main()
