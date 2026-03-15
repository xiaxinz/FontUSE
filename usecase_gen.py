import os
import json
import base64
from pathlib import Path

from openai import OpenAI

# =========================
# API key & client
# =========================
OPENAI_API_KEY = "YOUR_API_KEY"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# =========================
# Configuration
# =========================

dataset_name = "testdata-3"

DEFAULT_MODEL = "gpt-4o"

# Directory with input images (change this to your folder)
INPUT_DIR = "./testdata-3/resized-imgs"

# Directory where JSON outputs will be saved (change this to your folder)
OUTPUT_DIR = "./testdata-3/caption"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Paths to the saved prompts
SYSTEM_PROMPT_PATH = Path("system_prompt.txt")  # the system prompt I wrote for you
USER_PROMPT_PATH = Path("user_prompt.txt")      # the user prompt I wrote for you
# Load prompts
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
USER_PROMPT = USER_PROMPT_PATH.read_text(encoding="utf-8")
    
# Whether to split the usecases into 3 caption files per image
SPLIT_LONG_CAPTION = True  # set to False if you want only one caption per image

# =========================
# JSON schema for structured output
# =========================

FONT_SCHEMA_FORMAT = {
    "type": "json_schema",
    "name": "font_annotation",
    "schema": {
        "type": "object",
        "properties": {
            "suitable-for": {
                "type": "string",
                "description": "Short description of scenes/moods this font is suitable for."
            },
            "usecases": {
                "type": "array",
                "description": "3-4 concrete application scenes for this font.",
                "items": {"type": "string"}
            },
            "styles": {
                "type": "array",
                "description": "Font style keywords.",
                "items": {"type": "string"}
            },
            "colors": {
                "type": "array",
                "description": "Main colors of the font, or ['multiple'].",
                "items": {"type": "string"}
            },
        },
        "required": ["suitable-for", "usecases", "styles", "colors"],
        "additionalProperties": False,
    },
    "strict": True,
}

# =========================
# Helper functions
# =========================

def encode_image_to_data_url(image_path: Path) -> str:
    """Encode image file as a base64 data URL for vision input."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext == ".gif":
        mime = "image/gif"
    else:
        # Fallback mime type
        mime = "image/jpeg"

    return f"data:{mime};base64,{b64}"


def join_list(values) -> str:
    """Join a list of strings with '; '. Accepts list or single string."""
    if isinstance(values, list):
        return "; ".join(str(v) for v in values if v)
    if values is None:
        return ""
    return str(values)


def format_colors(colors) -> str:
    """
    Format colors field for caption.
    If it's ['multiple'] (case-insensitive), return empty string to skip.
    Otherwise join values with ', '.
    """
    if isinstance(colors, list):
        normalized = [str(c).strip() for c in colors if c is not None]
        if len(normalized) == 1 and normalized[0].lower() == "multiple":
            return ""
        return ", ".join(normalized)
    if colors is None:
        return ""
    s = str(colors).strip()
    if s.lower() == "multiple":
        return ""
    return s


def build_caption(suitable_for: str,
                  ideal_applications,
                  styles,
                  colors,
                  highlight_usecases: bool = False) -> str:
    """
    Build the final caption string.
    If highlight_usecases is True, wrap the usecases string in ** **.
    """
    usecases_str = join_list(ideal_applications)
    if highlight_usecases:
        usecases_str = f"**{usecases_str}**"

    styles_str = join_list(styles)
    colors_str = format_colors(colors)

    parts = [
        f"Suitable For: {suitable_for}",
        f"Ideal Applications: {usecases_str}",
        f"Font Styles: {styles_str}",
    ]
    if colors_str:
        parts.append(f"Colors: {colors_str}")

    return "; ".join(parts)


def split_usecases(usecases):
    """
    Split usecases into 3 segments based on rules:
    - If len >= 4: [first], [second], [rest...]
    - If len == 3: [0], [1], [2]
    - If len == 2: [0], [1], []
    - If len == 1: [0], [], []
    - If len == 0: [], [], []
    Returns a list of three lists: [part1, part2, part3].
    """
    if not isinstance(usecases, list):
        usecase_list = [usecases] if usecases else []
    else:
        usecase_list = [str(u) for u in usecases if u]

    n = len(usecase_list)

    if n >= 4:
        return [
            [usecase_list[0]],
            [usecase_list[1]],
            usecase_list[2:],
        ]
    elif n == 3:
        return [
            [usecase_list[0]],
            [usecase_list[1]],
            [usecase_list[2]],
        ]
    elif n == 2:
        return [
            [usecase_list[0]],
            [usecase_list[1]],
            [],
        ]
    elif n == 1:
        return [
            [usecase_list[0]],
            [],
            [],
        ]
    else:
        return [[], [], []]


def analyze_image(image_path: Path) -> dict:
    """
    Call the OpenAI Responses API with gpt-4o to analyze a single image.
    Returns a dict with keys: suitable-for, usecases, styles, colors.
    """
    data_url = encode_image_to_data_url(image_path)

    response = client.responses.create(
        model=MODEL,
        # System-level instructions
        instructions=SYSTEM_PROMPT,
        # User-level prompt + image
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "high",
                    },
                ],
            }
        ],
        # Structured Outputs: enforce JSON schema
        text={
            "format": FONT_SCHEMA_FORMAT
        },
        max_output_tokens=400,
    )

    # The text output should be a JSON string following our schema.
    raw_json = response.output_text
    return json.loads(raw_json)


# =========================
# Main processing loop
# =========================

def process_images(input_dirs=INPUT_DIR, output_dirs=OUTPUT_DIR):
    """Process all images in INPUT_DIR and save captions to OUTPUT_DIR."""
    supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    
    for INPUT_DIR, OUTPUT_DIR in zip(input_dirs, output_dirs):

        for image_path in sorted(INPUT_DIR.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in supported_exts:
                continue

            stem = image_path.stem

            if SPLIT_LONG_CAPTION:
                expected_files = [
                    OUTPUT_DIR / f"{stem}_caption_1.json",
                    OUTPUT_DIR / f"{stem}_caption_2.json",
                    OUTPUT_DIR / f"{stem}_caption_3.json",
                ]
            else:
                expected_files = [OUTPUT_DIR / f"{stem}_caption.json"]

            # Skip this image if ANY of the expected output files already exist.
            # This makes it easy to resume after interruption.
            if any(p.exists() for p in expected_files):
                print(f"[SKIP] {image_path.name} (some output already exists)")
                continue

            print(f"[PROCESS] {image_path.name}")
            try:
                annotation = analyze_image(image_path)
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")
                continue

            suitable_for = annotation.get("suitable-for", "")
            usecases = annotation.get("usecases", [])
            styles = annotation.get("styles", [])
            colors = annotation.get("colors", [])

            if SPLIT_LONG_CAPTION:
                # Split usecases into 3 parts and write 3 caption files.
                parts = split_usecases(usecases)

                for idx, part in enumerate(parts, start=1):
                    caption = build_caption(
                        suitable_for=suitable_for,
                        ideal_applications=part,
                        styles=styles,
                        colors=colors,
                        highlight_usecases=True,
                    )

                    out_data = {
                        "img_name": image_path.name,
                        "caption": caption,
                        "suitable-for": suitable_for,
                        "usecases": usecases,
                        "styles": styles,
                        "colors": colors,
                    }

                    out_path = OUTPUT_DIR / f"{stem}_caption_{idx}.json"
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(out_data, f, ensure_ascii=False, indent=2)

                print(f"  Saved 3 caption files for {image_path.name}")

            else:
                # Single caption per image.
                caption = build_caption(
                    suitable_for=suitable_for,
                    ideal_applications=usecases,
                    styles=styles,
                    colors=colors,
                    highlight_usecases=False,
                )

                out_data = {
                    "img_name": image_path.name,
                    "caption": caption,
                    "suitable-for": suitable_for,
                    "usecases": usecases,
                    "styles": styles,
                    "colors": colors,
                }

                out_path = OUTPUT_DIR / f"{stem}_caption.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(out_data, f, ensure_ascii=False, indent=2)

                print(f"  Saved caption file for {image_path.name}")


if __name__ == "__main__":
    process_images(INPUT_DIR, OUTPUT_DIR)
