# %%
from PIL import ImageFile
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv

from .config import GENERATED_VISUALS_DIR, PROJECT_ROOT
from .utils import hf_hub_login


# %%
def save_original_visuals():
    pass


# %%
if __name__ == "__main__":
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    hf_hub_login()

    ds: DatasetDict = load_dataset(
        "junling24/Math2Visual-Generating_Pedagogically_Meaningful_Visuals_for_Math_Word_Problems"
    )

    img: ImageFile.ImageFile = ds["train"][0]["image"]
    img.show()
