"""Download RNA-FM pretrained weights with retry for flaky CUHK server."""
import os
import time
import urllib.request

# Try own HuggingFace mirror first (fast CDN), then CUHK (slow, flaky)
URLS = [
    "https://huggingface.co/orgava/rna-fm-weights/resolve/main/RNA-FM_pretrained.pth",
    "https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth",
]
DST = os.path.expanduser("~/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth")
MIN_SIZE = 1_000_000_000  # 1GB

os.makedirs(os.path.dirname(DST), exist_ok=True)

for url in URLS:
    print(f"Trying: {url[:60]}...")
    for i in range(10):
        try:
            print(f"  Attempt {i+1}/10...")
            urllib.request.urlretrieve(url, DST)
            sz = os.path.getsize(DST)
            if sz > MIN_SIZE:
                print(f"OK: {sz / 1e6:.0f} MB from {url[:40]}")
                exit(0)
            print(f"  Incomplete: {sz / 1e6:.0f} MB")
        except Exception as e:
            print(f"  Failed: {e}")
        time.sleep(5)
    print(f"All attempts failed for {url[:40]}, trying next source...")

print("WARNING: RNA-FM download incomplete from all sources")
