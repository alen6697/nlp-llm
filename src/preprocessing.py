import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Basic text cleaning for clinical notes.
    """

    text = text.lower()

    text = re.sub(r"\n", " ", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, text_column="text") -> pd.DataFrame:
    """
    Apply preprocessing to the dataframe.
    """

    df[text_column] = df[text_column].astype(str).apply(clean_text)

    return df
