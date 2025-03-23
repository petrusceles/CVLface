import gdown
import argparse

# List of model URLs
model_url = [
    "https://drive.google.com/file/d/1NXeippshXSnFSD6uPZljXR71VOm1zdzw/view?usp=drive_link",
    "https://drive.google.com/file/d/1xPEvgkv2Unb8jIVUd8D4RRSvG0pH4zvb/view?usp=drive_link",
    "https://drive.google.com/file/d/1NNo1Pn3ZgmY6holl0Krk8z8Tv5dhyHWr/view?usp=drive_link",
    "https://drive.google.com/file/d/1k-G69R9S-zou5oAHPnBHs9aVi327VUDl/view?usp=drive_link",
    "https://drive.google.com/file/d/1kKAlvXBLURGkHhHnKy0kNDFeYa7X0BiY/view?usp=drive_link",
    "https://drive.google.com/file/d/1EOMFVz94v-qoTQdjkNrYgFu1QZGzY90F/view?usp=drive_link",
    "https://drive.google.com/file/d/1dWhoSOfBbT5WniWjvDDK_-kdCGBKM38-/view?usp=drive_link",
    "https://drive.google.com/file/d/1yQU_KDxQtTcCQx5SosP9xhfJrEfKynnB/view?usp=drive_link",
    "https://drive.google.com/file/d/150U99wcomv_IeON7Q1QEWethQJ8wOyTa/view?usp=drive_link",
    "https://drive.google.com/file/d/1c6x4Ql4YhR_ne06rgHfGsOGqzXw2S551/view?usp=drive_link",
    "https://drive.google.com/file/d/1M84mQnI5iJ0lgujyWw3lJNtflQ_P4286/view?usp=drive_link",
    "https://drive.google.com/file/d/1obX7megQrdNJEv9GhMDefgI3wrTq1Ruk/view?usp=drive_link",
    "https://drive.google.com/file/d/1kRkCsxABtLlAvB6XxlMV5xY2hqL71byl/view?usp=drive_link",
    "https://drive.google.com/file/d/1vNCGItodA3rPf8FYeQr-p_CLVvF3meRv/view?usp=drive_link",
    "https://drive.google.com/file/d/1dggZsfr5tmS3KHitkgMrb5sapT5FDtoK/view?usp=drive_link",
    "https://drive.google.com/file/d/1v1W7U_eJNPA4Cfe9EPXkhPkB0y8q7Qly/view?usp=drive_link",
    "https://drive.google.com/file/d/13ZcYt3RUdDHd4WC6RweibPpRgPzuT56n/view?usp=drive_link",
    "https://drive.google.com/file/d/1kAQLBWaDcFQSh0y4-chlx9JEfoqs-iFT/view?usp=drive_link",
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
            str(f"model.pt"),
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
