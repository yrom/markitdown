from typing import BinaryIO, Union
import base64
import mimetypes
import hashlib
import io
from .._stream_info import StreamInfo

MEMCACHE = {}


def encode_image_as_base64_data_uri(image_data: bytes, max_short_side_length: int = 1080) -> str:
    from PIL import Image

    image = Image.open(io.BytesIO(image_data), mode="r", formats=["JPEG", "JPG", "PNG"])
    if min(image.size) <= max_short_side_length and image.format in ["JPEG", "JPG"]:
        return "data:image/jpeg;base64," + base64.b64encode(image_data).decode("utf-8")

    if (max_short_side_length > 0) and (min(image.size) > max_short_side_length):
        ori_size = image.size
        image = resize_image(image, short_side_length=max_short_side_length)
        print(f"Image resized from {ori_size} to {image.size}.")

    image = image.convert(mode="RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def resize_image(img, short_side_length: int = 1080):
    from PIL import Image

    assert isinstance(img, Image.Image)

    width, height = img.size

    if width <= height:
        new_width = short_side_length
        new_height = int((short_side_length / width) * height)
    else:
        new_height = short_side_length
        new_width = int((short_side_length / height) * width)

    resized_img = img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    return resized_img


def llm_caption(file_stream: BinaryIO, stream_info: StreamInfo, *, client, model, prompt=None) -> Union[None, str]:
    if prompt is None or prompt.strip() == "":
        prompt = "Write a detailed caption for this image. Use the main language identified in the picture."

    # Get the content type
    content_type = stream_info.mimetype
    if not content_type:
        content_type, _ = mimetypes.guess_type("_dummy" + (stream_info.extension or ""))
    if not content_type:
        content_type = "application/octet-stream"

    # Convert to base64
    cur_pos = file_stream.tell()
    try:
        file_data = file_stream.read()
        cache_key = hashlib.md5(file_data).hexdigest()
        if cache_key in MEMCACHE:
            return MEMCACHE[cache_key]
        data_uri = encode_image_as_base64_data_uri(file_data, max_short_side_length=1080)
    except Exception:
        return None
    finally:
        file_stream.seek(cur_pos)

    # Prepare the data-uri
    # data_uri = f"data:{content_type};base64,{base64_image}"

    # Prepare the OpenAI API request
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri,
                    },
                },
            ],
        }
    ]

    # Call the OpenAI API
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.6)
    caption = response.choices[0].message.content
    # Cache the result
    MEMCACHE[cache_key] = caption
    return caption
