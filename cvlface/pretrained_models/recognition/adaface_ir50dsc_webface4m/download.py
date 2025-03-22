import gdown
import argparse

# List of model URLs
model_url = [
    "https://drive.google.com/file/d/1mvCBwjcMOwbn_FYJUrtEYtcWCRvaeBCv/view?usp=drive_link",
    "https://drive.google.com/file/d/1aydsnEEyrevKcyycTpeHdkX8-6L8ysDT/view?usp=drive_link",
    "https://drive.google.com/file/d/1o2Xd2O_Vd39RrpaQ0k8YvzSuRYuXJ0Wy/view?usp=drive_link",
    "https://drive.google.com/file/d/1yA3Tof02dHmgxpTR2RzabrMBjBL-r_Ob/view?usp=drive_link",
    "https://drive.google.com/file/d/1Sknlgc2Pu3bUDEose3_MzlQmKRZfQ1nX/view?usp=drive_link",
    "https://drive.google.com/file/d/1WXrQXjXS9woWbynaRPn_oya9XqBo6aqu/view?usp=drive_link",
    "https://drive.google.com/file/d/1W_Kz6BCXFT5xTQYhzDdKCciM-IUc3PqA/view?usp=drive_link",
    "https://drive.google.com/file/d/1WLxPUWg4M7evYQVVIdKg6UzYLVMqg80P/view?usp=drive_link",
    "https://drive.google.com/file/d/1--tsEK-ZiGoek_VHR-F1WTxyItp-J7oY/view?usp=drive_link",
    "https://drive.google.com/file/d/1P2T27ZABWHI7JaDNnWkkFI_zMAXnzQta/view?usp=drive_link",
    "https://drive.google.com/file/d/1TZRBJIJmLO-7bUZ6SDLBEFxTDo_5sVnH/view?usp=drive_link",
    "https://drive.google.com/file/d/1AvPeqpgPnuLkJQRse0q3Gk9qLVahraEk/view?usp=drive_link",
    "https://drive.google.com/file/d/1BsfmzDyoMpYDrRKKH2IA62R0UIfEv0xw/view?usp=drive_link",
    "https://drive.google.com/file/d/1apda1b34SoG-0I7zzsXWVOWbJGD6GTAl/view?usp=drive_link",
    "https://drive.google.com/file/d/1ftocqZieqymJ6fwk0QL7WfJfDko_V_IU/view?usp=drive_link",
    "https://drive.google.com/file/d/1HSlEUELkriLRU4I9w5pArDfZQ_n6_C7G/view?usp=drive_link",
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
