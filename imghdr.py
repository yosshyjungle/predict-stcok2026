from __future__ import annotations
from typing import Optional
from PIL import Image
import io


def what(file: Optional[str] = None, h: Optional[bytes] = None) -> Optional[str]:
    """Minimal replacement for the removed stdlib `imghdr.what` using Pillow.

    - If `h` (bytes) is provided, detect from the bytes buffer.
    - Otherwise `file` should be a filesystem path to open.
    Returns a lowercase format string like 'jpeg', 'png', or None if unknown.
    """
    try:
        if h is not None:
            buf = io.BytesIO(h)
            img = Image.open(buf)
        else:
            if not file:
                return None
            with open(file, "rb") as f:
                img = Image.open(f)
        fmt = img.format
        if not fmt:
            return None
        return fmt.lower()
    except Exception:
        return None
