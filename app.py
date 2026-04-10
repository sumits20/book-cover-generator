import os
import io
import base64
import random
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
    page_title="Book Cover Generator",
    page_icon="📚",
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


def decode_base64_to_pil(b64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))


def fetch_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def normalize_cover_prompt(
    title: str,
    subtitle: str,
    theme: str,
    age_group: str,
    extra_style: str,
    include_text: bool,
) -> str:
    text_rule = (
        f'Include the exact title "{title}" and subtitle "{subtitle}" in a clean, readable, professional children’s book cover layout.'
        if include_text
        else "Do not render any text or letters in the image. Leave natural empty composition space for later text placement."
    )

    prompt = f"""
Create a professional children's book cover illustration.
Theme: {theme}
Target age group: {age_group}
Book title: {title}
Subtitle: {subtitle}

Style requirements:
- vibrant full color illustration
- cartoonish, cheerful, happy, kid-friendly style
- soft lighting with cinematic depth of field
- same color tone and warm visual temperature across matching covers
- rich colors with smooth gradients
- professional children’s publishing quality
- detailed but clean composition
- appealing foreground subject with dimensional depth
- polished storybook illustration feel
- visually cohesive series-style artwork

Consistency requirements:
- maintain the same visual identity across front and back
- keep the same palette family, lighting mood, color temperature, and world design
- matching illustration style across both images

{text_rule}

Additional styling:
{extra_style}
""".strip()

    return prompt


# =========================
# Provider functions
# =========================
def generate_with_openai(prompt: str, size: str, model: str) -> Tuple[Image.Image, Optional[str], str]:
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
    image_url = getattr(item, "url", None)

    if getattr(item, "b64_json", None):
        img = decode_base64_to_pil(item.b64_json)
    elif image_url:
        img = fetch_image_from_url(image_url)
    else:
        raise RuntimeError("OpenAI returned no image data.")

    return img, image_url, "Generated with OpenAI"


def generate_with_grok(prompt: str, model: str) -> Tuple[Image.Image, Optional[str], str]:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Add it to requirements.txt.")
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY is missing.")

    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    result = client.images.generate(
        model=model,
        prompt=prompt,
    )

    item = result.data[0]
    image_url = getattr(item, "url", None)

    if getattr(item, "b64_json", None):
        img = decode_base64_to_pil(item.b64_json)
    elif image_url:
        img = fetch_image_from_url(image_url)
    else:
        raise RuntimeError("Grok/xAI returned no image data.")

    return img, image_url, "Generated with Grok / xAI"


def generate_with_together(
    prompt: str,
    model: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    reference_images: Optional[list[str]] = None,
) -> Tuple[Image.Image, Optional[str], str]:
    if Together is None:
        raise RuntimeError("together package is not installed. Add it to requirements.txt.")
    if not TOGETHER_API_KEY:
        raise RuntimeError("TOGETHER_API_KEY is missing.")

    client = Together(api_key=TOGETHER_API_KEY)

    kwargs = {
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "n": 1,
    }

    # Only use reference images for models that support them
    reference_capable_models = [
        "black-forest-labs/FLUX.1-Kontext-pro",
        "black-forest-labs/FLUX.1-Kontext-max",
    ]

    if reference_images and model in reference_capable_models:
        kwargs["reference_images"] = reference_images

    response = client.images.generate(**kwargs)

    item = response.data[0]
    image_url = getattr(item, "url", None)

    if getattr(item, "b64_json", None):
        img = decode_base64_to_pil(item.b64_json)
    elif image_url:
        img = fetch_image_from_url(image_url)
    else:
        raise RuntimeError("Together AI returned no image data.")

    note = f"Generated with Together AI using {model} (seed={seed})"
    if reference_images and model not in reference_capable_models:
        note += " | reference image skipped because this model does not support it"

    return img, image_url, note

# =========================
# Prompt builders
# =========================
def build_front_prompt(base_prompt: str) -> str:
    return base_prompt + """

Image goal:
- FRONT COVER
- main hero subject clearly visible
- strong focal point
- attractive foreground composition
- premium children’s book cover feel
- leave some clean space near the top for title placement
- keep it lively, joyful, and highly engaging
"""


def build_back_prompt(base_prompt: str) -> str:
    return base_prompt + """

Image goal:
- BACK COVER
- background-only style scene for blurb placement
- same world, same theme, same palette family, same lighting mood, same temperature as the front
- NO main character portrait
- NO central subject
- NO large foreground figure
- keep decorative environmental details only
- provide a calmer composition with open readable space in the middle and upper-middle for blurb text
- suitable as a professional back cover background
- No texts
"""


# =========================
# Dual generation
# =========================
def generate_dual_images(
    provider: str,
    base_prompt: str,
    model: str,
    size: Optional[str] = None,
    steps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
):
    front_prompt = build_front_prompt(base_prompt)
    back_prompt = build_back_prompt(base_prompt)

    if provider == "Together AI":
        if seed is None:
            raise RuntimeError("Seed is required for Together AI.")

        front_img, front_url, front_note = generate_with_together(
            prompt=front_prompt,
            model=model,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            reference_images=None,
        )

        back_img, back_url, back_note = generate_with_together(
            prompt=back_prompt,
            model=model,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            reference_images=[front_url] if front_url else None,
        )

        return {
            "front_image": front_img,
            "back_image": back_img,
            "front_url": front_url,
            "back_url": back_url,
            "note": f"{front_note}. Back cover used same seed and front reference image."
        }

    if provider == "OpenAI":
        front_img, front_url, front_note = generate_with_openai(
            prompt=front_prompt,
            size=size,
            model=model,
        )
        back_img, back_url, back_note = generate_with_openai(
            prompt=back_prompt,
            size=size,
            model=model,
        )
        return {
            "front_image": front_img,
            "back_image": back_img,
            "front_url": front_url,
            "back_url": back_url,
            "note": f"{front_note}. OpenAI version uses matched prompts but not shared seed/reference in this app."
        }

    if provider == "Grok / xAI":
        front_img, front_url, front_note = generate_with_grok(
            prompt=front_prompt,
            model=model,
        )
        back_img, back_url, back_note = generate_with_grok(
            prompt=back_prompt,
            model=model,
        )
        return {
            "front_image": front_img,
            "back_image": back_img,
            "front_url": front_url,
            "back_url": back_url,
            "note": f"{front_note}. xAI version uses matched prompts but not shared seed/reference in this app."
        }

    raise RuntimeError(f"Unsupported provider: {provider}")


# =========================
# UI
# =========================
st.title("📚 Front + Back Book Cover Generator")
st.caption("Generate a matching front cover and a back-cover background image for children's books.")

with st.sidebar:
    st.header("Provider")

    provider = st.selectbox(
        "Choose image provider",
        ["Together AI", "OpenAI", "Grok / xAI"],
        index=0,
    )

    if provider == "Together AI":
        provider_model = st.selectbox(
            "Together AI image model",
            [
                "black-forest-labs/FLUX.1-schnell",
                "black-forest-labs/FLUX.1-dev",
                "black-forest-labs/FLUX.1-Kontext-pro",
            ],
            index=0,
        )
        width = st.selectbox("Width", [768, 1024], index=1)
        height = st.selectbox("Height", [1024, 1536], index=1)
        together_steps = st.slider("Inference steps", min_value=2, max_value=20, value=8)
        use_random_seed = st.checkbox("Use random seed", value=True)

        if use_random_seed:
            seed_value = random.randint(1, 999999999)
            st.caption(f"Seed for this run: `{seed_value}`")
        else:
            seed_value = st.number_input("Seed", min_value=1, max_value=999999999, value=12345, step=1)

        image_size = None

    elif provider == "OpenAI":
        provider_model = st.selectbox(
            "OpenAI image model",
            ["gpt-image-1"],
        )
        image_size = st.selectbox(
            "Image size",
            ["1024x1024", "1024x1536", "1536x1024"],
            index=1,
        )
        width = None
        height = None
        together_steps = None
        seed_value = None

    else:
        provider_model = st.selectbox(
            "xAI image model",
            ["grok-imagine-image"],
        )
        image_size = None
        width = None
        height = None
        together_steps = None
        seed_value = None

    st.divider()
    st.subheader("API key status")
    st.write(f"Together AI: {'✅ Found' if TOGETHER_API_KEY else '❌ Missing'}")
    st.write(f"OpenAI: {'✅ Found' if OPENAI_API_KEY else '❌ Missing'}")
    st.write(f"xAI / Grok: {'✅ Found' if XAI_API_KEY else '❌ Missing'}")

    st.divider()
    st.info(
        "Best consistency path: Together AI + same seed + front image reused as reference for back cover."
    )

col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("Book details")

    title = st.text_input("Book title", value="Celebration")
    subtitle = st.text_input("Subtitle", value="Fun Colouring Book for Kids Ages 4-10")
    theme = st.text_area(
        "Theme / subject",
        value="joyful children celebrating at a festive party with balloons, bunting, confetti, fireworks style decorations, warm happy cartoon atmosphere",
        height=120,
    )
    age_group = st.selectbox(
        "Age group",
        ["Ages 3-5", "Ages 4-8", "Ages 4-10", "Ages 6-9", "Ages 8-12"],
        index=2,
    )
    include_text = st.checkbox(
        "Ask model to include title text in image",
        value=False,
        help="Usually better to keep this off and add text later in Canva or Photoshop.",
    )
    extra_style = st.text_area(
        "Extra style instructions",
        value="warm festive palette, soft sunlight feel, cheerful expressions, premium kids book illustration, clean readable composition, polished cartoon style, joyful and bright",
        height=120,
    )

    manual_prompt = st.text_area(
        "Optional: override prompt manually",
        value="",
        height=180,
        placeholder="Leave blank to use the app-generated prompt.",
    )

    if st.button("Generate Front + Back", type="primary", use_container_width=True):
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

            with st.spinner("Generating matching covers..."):
                result = generate_dual_images(
                    provider=provider,
                    base_prompt=final_prompt,
                    model=provider_model,
                    size=image_size,
                    steps=together_steps,
                    width=width,
                    height=height,
                    seed=seed_value,
                )

            st.session_state["front_image"] = result["front_image"]
            st.session_state["back_image"] = result["back_image"]
            st.session_state["front_url"] = result["front_url"]
            st.session_state["back_url"] = result["back_url"]
            st.session_state["generation_note"] = result["note"]
            st.session_state["seed_used"] = seed_value

            st.success("Front cover and back-cover background generated successfully.")

        except Exception as e:
            st.error(f"Generation failed: {e}")

with col2:
    st.subheader("Preview")

    if "front_image" in st.session_state and "back_image" in st.session_state:
        prev1, prev2 = st.columns(2)

        with prev1:
            st.image(st.session_state["front_image"], caption="Front Cover", use_container_width=True)
            st.download_button(
                "Download Front PNG",
                data=pil_to_png_bytes(st.session_state["front_image"]),
                file_name="front_cover.png",
                mime="image/png",
                use_container_width=True,
            )

        with prev2:
            st.image(st.session_state["back_image"], caption="Back Cover Background", use_container_width=True)
            st.download_button(
                "Download Back PNG",
                data=pil_to_png_bytes(st.session_state["back_image"]),
                file_name="back_cover_background.png",
                mime="image/png",
                use_container_width=True,
            )

        st.caption(st.session_state.get("generation_note", ""))

        if st.session_state.get("seed_used") is not None:
            st.caption(f"Seed used: `{st.session_state['seed_used']}`")

    else:
        st.info("Your generated front cover and back-cover background will appear here.")

st.divider()

with st.expander("Show final prompt", expanded=False):
    st.code(st.session_state.get("final_prompt", "Generate once to see the final prompt here."), language="text")

with st.expander("Secrets for Streamlit Cloud", expanded=False):
    st.code(
        '''
OPENAI_API_KEY = "your_openai_key"
XAI_API_KEY = "your_xai_key"
TOGETHER_API_KEY = "your_together_key"
'''.strip(),
        language="toml",
    )

with st.expander("requirements.txt", expanded=False):
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
- Together AI is the recommended provider here for strongest cover-to-cover consistency.
- For Together, this app keeps the same seed and reuses the front image as a reference for the back image.
- The back cover prompt is intentionally written as a background-only composition so you can place your own blurb later.
- For OpenAI and xAI, this app still generates both images, but without the same seed/reference workflow used for Together.
"""
)
