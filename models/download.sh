#!/usr/bin/env bash
# Fetch model weights into models/ (they are not committed). Checksums are in manifest.json.
set -euo pipefail
cd "$(dirname "$0")"
curl -fL -o face_detection_yunet_2026may.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx
for v in T S B; do
  curl -fL -o "LVFace-${v}_Glint360K.onnx" "https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-${v}_Glint360K/LVFace-${v}_Glint360K.onnx"
done
sha256sum *.onnx
