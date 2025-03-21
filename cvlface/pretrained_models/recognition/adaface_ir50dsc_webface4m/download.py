import gdown
import argparse

# List of model URLs
model_url = [
    "https://drive.google.com/file/d/1EsOdwh6Whv0DNqicmsIZPvtsCeq13jQb/view?usp=drive_link",
    "https://drive.google.com/file/d/1MFcHtdPkNhxWbu8klyxqmMQ2ObxD0IBt/view?usp=drive_link",
    "https://drive.google.com/file/d/1SlmccGsNnGg_s6bpTTq2lzNmDUQvXDtR/view?usp=drive_link",
    "https://drive.google.com/file/d/1NUKr6eR0AbROyzxtRzlNjUFadNq1Qujw/view?usp=drive_link",
    "https://drive.google.com/file/d/14RP77JfWIKJHwknIH1KTyFglgC1NWnPK/view?usp=drive_link",
    "https://drive.google.com/file/d/13wwKVqDeAaoShRk3Y222xe2a0fTYiwWV/view?usp=drive_link",
    "https://drive.google.com/file/d/1I-sq6JK6XyTUOrJPnmF7uO7W1DMghkYK/view?usp=drive_link",
    "https://drive.google.com/file/d/1uCkcCmBgUkvgDvg3-N-fShByt94LBlcJ/view?usp=drive_link",
    "https://drive.google.com/file/d/1EZ190l-7_XWWueiIHpfnP-Pmddpt12ZH/view?usp=drive_link",
    "https://drive.google.com/file/d/1oXpk_bi8Ha-vogwyfG_0_F7-LuPSUdb6/view?usp=drive_link",
    "https://drive.google.com/file/d/1bni8urI8n1Yy8EOcxfdYyVyE0QQNTPWB/view?usp=drive_link",
    "https://drive.google.com/file/d/1W0iylHfgT3E-Jha6b_GHJkVVvzdNgwQs/view?usp=drive_link",
    "https://drive.google.com/file/d/1ZTVfLSbsGLEhM4n-toK0jNzRUNiDjXfd/view?usp=drive_link",
    "https://drive.google.com/file/d/1e2axAOSSpe33_kEIZ7yZfk6n2gGttGZ4/view?usp=drive_link",
    "https://drive.google.com/file/d/1jvZYp5mt-WNo8papO9-6_WW_qOLJJETg/view?usp=drive_link",
    "https://drive.google.com/file/d/1Wgamc6SNgXMTQrHSu1QxwTcn7Wn0C9o-/view?usp=drive_link",
    "https://drive.google.com/file/d/1fUGGvWfS-FX6G_HmizXhvyBL7OiYUjIz/view?usp=drive_link",
    "https://drive.google.com/file/d/14Dl7cuULOvqbG2BQwZCqfaoW2vFaAk3O/view?usp=drive_link",
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
