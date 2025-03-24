import gdown
import argparse

# List of model URLs
model_url = [
    "https://drive.google.com/file/d/1tvEevbSRtvekiZjGf58NCSy-ycI9AA3o/view?usp=drive_link",
    "https://drive.google.com/file/d/1trNd3WFZTXVpY3s_gBgeKCl3cgvFVtgo/view?usp=drive_link",
    "https://drive.google.com/file/d/1ibB_oytc3LGiWtwmVoyEQsFjdn-Wf3Cn/view?usp=drive_link",
    "https://drive.google.com/file/d/1htSk_FpKw70IRemnqI7JzGnSYqp3HLUK/view?usp=drive_link",
    "https://drive.google.com/file/d/1-DiCcLqSoVDITrB7WzX8RQn-1JYupYA0/view?usp=drive_link",
    "https://drive.google.com/file/d/16mhhFTUzXS6LffN2srBRuMdO3YivMUCn/view?usp=drive_link",
    "https://drive.google.com/file/d/19me785-9uzxidncckF12JOjR9vFZsk-X/view?usp=drive_link",
    "https://drive.google.com/file/d/1mfmPHCg4sKxh3Kf3i53pYDMpaCa80_7C/view?usp=drive_link",
    "https://drive.google.com/file/d/151OvOujt8v7Bl2AdPAgYE6GvavrqTYzn/view?usp=drive_link",
    "https://drive.google.com/file/d/1hZ36HAsNcDNEglH_OLILo1ClNlDwUSOH/view?usp=drive_link",
    "https://drive.google.com/file/d/19arOCGBCUQsZzJ5jbWcDn5qsJlOCLyag/view?usp=drive_link",
    "https://drive.google.com/file/d/1UNgk8DLzLq6zCMaMXAfPyxXNtBow_Nk_/view?usp=drive_link",
    "https://drive.google.com/file/d/1Fad8JDsJTKg9QpUByUt4YkNGN-ZftD3U/view?usp=drive_link",
    "https://drive.google.com/file/d/1oytFnzyNbDap5wvngfva9EtOuhfm6sCP/view?usp=drive_link",
    "https://drive.google.com/file/d/1TKVRQ5lmFqAMuvBSBumkmU2T8BU-fJ72/view?usp=drive_link",
    "https://drive.google.com/file/d/1szt556w2guJPcEUtPyNvpBGXxNIUcJuj/view?usp=drive_link",
    "https://drive.google.com/file/d/1Ck7yi5kDrC_b_69MpOLSoydnw3DGqfyR/view?usp=drive_link",
    "https://drive.google.com/file/d/1B_5dHvb73l0RqsjE_r-Tpb5M7PxVQKnD/view?usp=drive_link",
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
