import gdown
import argparse

# List of model URLs
model_url = [
    "https://drive.google.com/file/d/1YyPlbY55n9RZTW_zPICJ6hkX2YmUAdxH/view?usp=drive_link",
    "https://drive.google.com/file/d/12AP5-ciZbeq-Dm1Aoyhg8v23AqDraBGI/view?usp=drive_link",
    "https://drive.google.com/file/d/1FdZe1N-xod8D2tBiqEjoV3kr_BmJCQr0/view?usp=drive_link",
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
