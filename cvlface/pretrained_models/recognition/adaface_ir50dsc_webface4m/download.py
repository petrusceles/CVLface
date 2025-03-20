import gdown

24 - 11
model_url = [
    "https://drive.google.com/file/d/12jqSFyE9qMbNsujUiX0WAjHOXOAh43HK/view?usp=drive_link",
    "https://drive.google.com/file/d/1go-EsJqYmj6G1FVvGs0OfiqFTRATJuyu/view?usp=drive_link",
    "https://drive.google.com/file/d/1Uiuw2xBrqYnsYqPx1qaKGJJEPvlmT3x5/view?usp=drive_link",
    "https://drive.google.com/file/d/1otQHcT2DNndviwjRfjyAdz2HzEA_Zz7n/view?usp=drive_link",
    "https://drive.google.com/file/d/1bP6Sxg1fIBlRAMQydyUzGCtE1-kVtYOW/view?usp=drive_link",
    "https://drive.google.com/file/d/1spQUgh0Xz3O8MMX4Kzd53PtfjCca28Zj/view?usp=drive_link",
    "https://drive.google.com/file/d/1pdXcvoFgWNlovbvOskxFg78eLG2yoC4E/view?usp=drive_link",
    "https://drive.google.com/file/d/1K2LltNIOvAi_3EIEDZydXwtSbj6OcuGv/view?usp=drive_link",
    "https://drive.google.com/file/d/1SZJX1rfalIpzrufkxiv0PcMBWGLyvOMK/view?usp=drive_link",
    "https://drive.google.com/file/d/1G3XAkeKa-OikKdTviQTkA-Hgz8nDZOjk/view?usp=drive_link",
    "https://drive.google.com/file/d/1pLGunUjwjQID4vMwiuqunomr9QIXiugl/view?usp=drive_link",
    "https://drive.google.com/file/d/1YFZvo8yqGt9z-kh_mSswkZdwGHHFZAF1/view?usp=drive_link",
    "https://drive.google.com/file/d/1YIH_hEne_IK7cCea4r-iVzvG-dFiG1BM/view?usp=drive_link",
    "https://drive.google.com/file/d/1I2-6pInZe6pGOxTCwGggqT2KDN6xQpGn/view?usp=drive_link",
]


gdown.download(model_url[0], str("model.pt"), quiet=False, fuzzy=True)
