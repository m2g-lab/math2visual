# %%
import huggingface_hub
import os

from typing import Optional


# %%
def hf_hub_login() -> None:
    hf_token: Optional[str] = os.getenv("HF_HUB_ACCESS_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token)
