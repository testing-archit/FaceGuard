"""Reversible redaction: keep the hidden pixels, encrypted, next to the export.

Some users need to hide faces for publication but still be able to undo it
later (an investigator, an archive). The redacted image is exported as
usual; the *original pixels under every mask* are cropped, compressed
losslessly (PNG) and encrypted into a separate `.fgvault` file. Without the
passphrase the vault reveals nothing but the number and position of boxes.

Cryptography (standard primitives only, no home-made crypto):
- Key derivation: scrypt (RFC 7914, `hashlib.scrypt`), a memory-hard KDF,
  so guessing passphrases costs memory as well as time. Random 16-byte salt.
- Encryption: AES-256-GCM (authenticated encryption, `cryptography`
  package). A wrong passphrase or any tampering with the vault makes
  decryption fail instead of returning garbage.
- The SHA-256 of the redacted image is bound in as associated data, so a
  vault only opens against the exact image it was made for.

Images only. A video vault would need every masked crop of every frame and
a lossless re-encode to restore exactly; that is left as future work (D-035).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from core.errors import FaceGuardError
from core.redact import build_mask
from core.types import Box, Config

VAULT_VERSION = 1
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**15, 8, 1  # ~32 MB and ~0.1 s per derivation
SALT_BYTES, NONCE_BYTES, KEY_BYTES = 16, 12, 32
MIN_PASSPHRASE = 8


def _aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # optional dependency: only the vault needs it
        raise FaceGuardError("Reversible redaction needs the `cryptography` package: pip install cryptography") from exc
    return AESGCM(key)


def _key(passphrase: str, salt: bytes) -> bytes:
    if len(passphrase) < MIN_PASSPHRASE:
        raise FaceGuardError(f"Vault passphrase must be at least {MIN_PASSPHRASE} characters.")
    return hashlib.scrypt(passphrase.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                          maxmem=2**26, dklen=KEY_BYTES)


def image_digest(img: np.ndarray) -> str:
    """SHA-256 of the decoded pixels (not the file), so PNG metadata cannot matter."""
    return hashlib.sha256(np.ascontiguousarray(img).tobytes() + str(img.shape).encode()).hexdigest()


def mask_regions(boxes: list[Box], w: int, h: int, cfg: Config) -> list[Box]:
    """The pixel rectangles a redaction can have changed: each mask's ROI."""
    return [build_mask(b, w, h, cfg)[0] for b in boxes if b.area > 0]


def seal(original: np.ndarray, redacted: np.ndarray, regions: list[Box], passphrase: str) -> bytes:
    """Encrypt the original pixels of `regions` (from `mask_regions`). Returns the vault file bytes."""
    buf = io.BytesIO()
    crops = {f"r{k}": original[r.y : r.y + r.h, r.x : r.x + r.w] for k, r in enumerate(regions)}
    np.savez_compressed(buf, **crops)
    payload = json.dumps({"regions": [r.as_list() for r in regions], "shape": list(original.shape)}).encode()
    plaintext = len(payload).to_bytes(4, "big") + payload + buf.getvalue()
    salt, nonce = os.urandom(SALT_BYTES), os.urandom(NONCE_BYTES)
    digest = image_digest(redacted)
    ct = _aesgcm(_key(passphrase, salt)).encrypt(nonce, plaintext, digest.encode())
    header = {"faceguard_vault": VAULT_VERSION, "kdf": "scrypt", "n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P,
              "cipher": "AES-256-GCM", "salt": base64.b64encode(salt).decode(),
              "nonce": base64.b64encode(nonce).decode(), "image_sha256": digest, "regions": len(regions),
              "ciphertext": base64.b64encode(ct).decode()}
    return json.dumps(header).encode()


def restore(redacted: np.ndarray, vault: bytes, passphrase: str) -> np.ndarray:
    """Paste the decrypted original pixels back into the redacted image."""
    try:
        header = json.loads(vault)
        if header.get("faceguard_vault") != VAULT_VERSION:
            raise ValueError
        salt, nonce = base64.b64decode(header["salt"]), base64.b64decode(header["nonce"])
        ct = base64.b64decode(header["ciphertext"])
    except (ValueError, KeyError, TypeError) as exc:
        raise FaceGuardError("Not a FaceGuard vault file.") from exc
    digest = image_digest(redacted)
    if digest != header["image_sha256"]:
        raise FaceGuardError("This vault belongs to a different image, or the exported file was re-encoded. "
                             "Restore needs the exact file FaceGuard exported.")
    key = hashlib.scrypt(passphrase.encode(), salt=salt, n=header["n"], r=header["r"], p=header["p"],
                         maxmem=2**26, dklen=KEY_BYTES)
    try:
        plaintext = _aesgcm(key).decrypt(nonce, ct, digest.encode())
    except Exception as exc:  # cryptography raises InvalidTag
        raise FaceGuardError("Wrong passphrase, or the vault file was modified.") from exc
    n = int.from_bytes(plaintext[:4], "big")
    meta = json.loads(plaintext[4 : 4 + n])
    crops = np.load(io.BytesIO(plaintext[4 + n :]))
    if list(redacted.shape) != meta["shape"]:
        raise FaceGuardError("Image size does not match the vault.")
    out = redacted.copy()
    for k, (x, y, w, h) in enumerate(meta["regions"]):
        out[y : y + h, x : x + w] = crops[f"r{k}"]
    return out


def vault_path(dst: str | Path) -> Path:
    return Path(str(dst) + ".fgvault")

