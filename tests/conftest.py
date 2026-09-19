"""Test-session configuration.

Jobs run inline (synchronously) in tests so API assertions can observe the
final state of a job. Production refuses to start unless JOB_EXECUTION is
'background' (see validate_production_config); background execution has its
own dedicated test.
"""

import io
import os

os.environ.setdefault("JOB_EXECUTION", "inline")


def png_bytes(text: str = "TEST", size=(240, 80)) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((10, 30), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
