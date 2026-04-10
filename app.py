import os
import io
import base64
from typing import Optional, Tuple

import requests
import streamlit as st
from PIL import Image

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from together import Together
except Exception:
    Together = None


st.set_page_config(
    page_title="Coloring Book Cover Generator",
    page_icon="🎨",
    layout="wide",
)


# =========================
# Helpers
# =========================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
XAI_API_KEY = get_secret("XAI_API_KEY")
TOGETHER_API_KEY = get_secret("TOGETHER_API_KEY")


def normalize_cover_prompt(
    title: str,
    subtitle: str,
    theme: str,
    age_group: str,
    extra_style: str,
    include_text: bool,
) -> str:
    text_rule = (
        "Include the exact title text and subtitle text in a clean, readable, centered book-cover layout."
        if include_text
        else "Do not render any text or letters in the image; create illustration-only cover art with a blank title area at the top."
    )

    prompt = f"""
Create a professional coloring book front cover.
Theme: {theme}
Target age group: {age_group}
Book title: {title}
Subtitle: {subtitle}
Style requirements:
- vibrant full color illustration
- cartoonish, cheerful, kid-friendly style
- soft lighting with depth of field (background slightly blurred)
- rich colors with smooth gradients
- detailed but clean composition
- professional children’s book cover style
- centered balanced composition
- visually engaging foreground subject with depth
- portrait front cover composition
- joyful and appealing mood
{text_rule}
Additional styling: {extra_style}
""".strip()
    return prompt


def decode_base64_to_pil(b64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))


def fetch_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =========================
# Provider functions
# =========================
def generate_with_openai(prompt: str, size: str, model: str) -> Tuple[Image.Image, str]:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Add it to requirements.txt.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    result = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
    )

    item = result.data[0]
    if getattr(item, "b64_json", None):
        img = decode_base64_to_pil(item.b64_json)
    elif getattr(item, "url", None):
        img = fetch_image_from_url(item.url)
    else:
        raise RuntimeError("OpenAI returned no image data.")

    return img, "Generated with OpenAI"


def generate_with_together(prompt: str, model: str, steps: int, n: int = 1) -> Tuple[Image.Image, str]:
    if Together is None:
        raise RuntimeError("together package is not installed. Add it to requirements.txt.")
    if not TOGETHER_API_KEY:
        raise RuntimeError("TOGETHER_API_KEY is missing.")

    client = Together(api_key=TOGETHER_API_KEY)
    response = client.images.generate(
        prompt=prompt,
        model=model,
        steps=steps,
        n=n,
    )

    item = response.data[0]
    if getattr(item, "b64_json", None):
        img = decode_base64_to_pil(item.b64_json)
    elif getattr(item, "url", None):
        img = fetch_image_from_url(item.url)
    else:
        raise RuntimeError("Together AI returned no image data.")

    return img, f"Generated with Together AI using {model}"


def generate_with_grok(prompt: str, model: str) -> Tuple[Image.Image, str]:
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY is missing.")

    # xAI supports OpenAI-compatible image generation at base_url=https://api.x.ai/v1
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Add it to requirements.txt.")

    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    result = client.images.generate(
        model=model,
        prompt=prompt,
    )

    item = result.data[0]
    if getattr(item, "b64_json", None):
        img = decode_base64_to_pil(item.b64_json)
    elif getattr(item, "url", None):
        img = fetch_image_from_url(item.url)
    else:
        raise RuntimeError("Grok/xAI returned no image data.")

    return img, "Generated with Grok / xAI"


# =========================
# Dual Image Generator (Front + Back)
# =========================
def generate_dual_images(provider, prompt, model, size=None, steps=None):
    """Generate two themed images: front and back cover"""

    front_prompt = prompt + "\nFocus on FRONT COVER: main subject centered, bold composition, title space at top."
    back_prompt = prompt + "\nFocus on BACK COVER: simpler layout, background scene, space for text blurb, minimal central subject."

    if provider == "OpenAI":
        front, _ = generate_with_openai(front_prompt, size, model)
        back, _ = generate_with_openai(back_prompt, size, model)

    elif provider == "Grok / xAI":
        front, _ = generate_with_grok(front_prompt, model)
        back, _ = generate_with_grok(back_prompt, model)

    else:
        front, _ = generate_with_together(front_prompt, model, steps)
        back, _ = generate_with_together(back_prompt, model, steps)

    return front, back


# =========================
# UI
# =========================
st.title("🎨 Coloring Book Front Page Generator")
st.caption("Generate printable coloring-book front cover art using OpenAI, Grok/xAI, or Together AI.")

with st.sidebar:
    st.header("Provider")
    provider = st.selectbox(
        "Choose image provider",
        ["OpenAI", "Grok / xAI", "Together AI"],
    )

    if provider == "OpenAI":
        provider_model = st.selectbox(
            "OpenAI image model",
            ["gpt-image-1"],
        )
        image_size = st.selectbox(
            "Image size",
            ["1024x1024", "1024x1536", "1536x1024"],
            index=1,
        )

    elif provider == "Grok / xAI":
        provider_model = st.selectbox(
            "xAI image model",
            ["grok-imagine-image"],
        )
        image_size = "portrait"

    else:
        provider_model = st.selectbox(
            "Together AI image model",
            [
                "black-forest-labs/FLUX.1-schnell",
                "black-forest-labs/FLUX.1-dev",
                "stabilityai/stable-diffusion-xl-base-1.0",
            ],
        )
        together_steps = st.slider("Inference steps", min_value=2, max_value=20, value=8)
        image_size = "provider-managed"

    st.divider()
    st.subheader("API key status")
    st.write(f"OpenAI: {'✅ Found' if OPENAI_API_KEY else '❌ Missing'}")
    st.write(f"xAI / Grok: {'✅ Found' if XAI_API_KEY else '❌ Missing'}")
    st.write(f"Together AI: {'✅ Found' if TOGETHER_API_KEY else '❌ Missing'}")


col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Cover details")
    title = st.text_input("Book title", value="Cute Space Animals")
    subtitle = st.text_input("Subtitle", value="Fun Coloring Book for Kids Ages 4-8")
    theme = st.text_area(
        "Theme / subject",
        value="adorable animals in space, planets, stars, rockets, friendly alien details",
        height=100,
    )
    age_group = st.selectbox(
        "Age group",
        ["Ages 3-5", "Ages 4-8", "Ages 6-9", "Ages 8-12"],
        index=1,
    )
    include_text = st.checkbox(
        "Ask model to include title text in image",
        value=False,
        help="Many image models still struggle with perfect text. Keeping this off is usually safer.",
    )
    extra_style = st.text_area(
        "Extra style instructions",
        value="decorative border, fun cover composition, stars around the edges, bold black outlines, no shading, no grayscale",
        height=100,
    )

    manual_prompt = st.text_area(
        "Optional: edit the final prompt manually",
        value="",
        height=180,
        placeholder="Leave blank to use the app-generated prompt.",
    )

    if st.button("Generate covers (Front + Back)", type="primary", use_container_width=True):
        try:
            final_prompt = manual_prompt.strip() or normalize_cover_prompt(
                title=title,
                subtitle=subtitle,
                theme=theme,
                age_group=age_group,
                extra_style=extra_style,
                include_text=include_text,
            )

            st.session_state["final_prompt"] = final_prompt

            with st.spinner("Generating front and back covers..."):
                front_img, back_img = generate_dual_images(
                    provider=provider,
                    prompt=final_prompt,
                    model=provider_model,
                    size=image_size if provider == "OpenAI" else None,
                    steps=together_steps if provider == "Together AI" else None,
                )

            st.session_state["front_image"] = front_img
            st.session_state["back_image"] = back_img
            st.success("Front and back covers generated successfully.")

        except Exception as e:
            st.error(f"Generation failed: {e}")

        except Exception as e:
            st.error(f"Generation failed: {e}")

with col2:
    st.subheader("Preview")

    if "front_image" in st.session_state:
        colf, colb = st.columns(2)

        with colf:
            st.image(st.session_state["front_image"], caption="Front Cover", use_container_width=True)
            front_bytes = pil_to_png_bytes(st.session_state["front_image"])
            st.download_button("Download Front", front_bytes, file_name="front_cover.png")

        with colb:
            st.image(st.session_state["back_image"], caption="Back Cover", use_container_width=True)
            back_bytes = pil_to_png_bytes(st.session_state["back_image"])
            st.download_button("Download Back", back_bytes, file_name="back_cover.png")

    else:
        st.info("Your generated covers will appear here.")

st.divider()

with st.expander("Show final prompt", expanded=True):
    st.code(st.session_state.get("final_prompt", "Generate once to see the final prompt here."), language="text")

with st.expander("Suggested secrets.toml"):
    st.code(
        '''
OPENAI_API_KEY = "your_openai_key"
XAI_API_KEY = "your_xai_key"
TOGETHER_API_KEY = "your_together_key"
'''.strip(),
        language="toml",
    )

with st.expander("Suggested requirements.txt"):
    st.code(
        '''
streamlit
openai
requests
pillow
together
'''.strip(),
        language="text",
    )

st.markdown(
    """
### Notes
- OpenAI image generation is available through the image generation API guide. ([platform.openai.com](https://platform.openai.com/docs/guides/image-generation))
- xAI supports image generation with `grok-imagine-image`, and also provides an OpenAI-compatible API pattern at `https://api.x.ai/v1`. ([docs.x.ai](https://docs.x.ai/developers/model-capabilities/images/generation))
- Together AI supports image generation through `client.images.generate(...)`, including models such as FLUX and SDXL. ([docs.together.ai](https://docs.together.ai/docs/images-overview))
- Streamlit recommends keeping API keys in `.streamlit/secrets.toml` or environment variables instead of hardcoding them. ([docs.streamlit.io](https://docs.streamlit.io/develop/concepts/connections/secrets-management))
"""
)
