# Benchmark

Date: 2026-09-23T17:07:24.241Z

- CPU: 13th Gen Intel(R) Core(TM) i9-13900K (32 logical cores)
- RAM: 31 GiB
- OS: Linux 6.8.0-139-generic x64
- Node: v24.19.0

## Detection (YuNet, CPU, inputLongSide=640)

| Frame | ms/frame | max fps (1 stream) |
|---|---|---|
| 1280×720 (4 faces) | 8.1 | 122.9 |
| 1920×1080 | 4.4 | 228.9 |

Alignment: 0.70 ms/face, quality metrics: 0.05 ms/face.

## Embeddings (LVFace)

| Model | EP | batch=1 ms | batch=5 ms | ms/face @5 |
|---|---|---|---|---|
| lvface-s-glint360k | cpu | 21.0 | 53.6 | 10.7 |
| lvface-s-glint360k | cuda | 1.6 | 3.4 | 0.7 |
| lvface-t-glint360k | cpu | 17.5 | 21.8 | 4.4 |
| lvface-t-glint360k | cuda | 1.2 | 2.1 | 0.4 |
| lvface-b-glint360k | cpu | 31.4 | 113.9 | 22.8 |
| lvface-b-glint360k | cuda | 3.9 | 5.7 | 1.1 |

## Matching (pure TS, brute force)

| Gallery | ms/probe |
|---|---|
| 10,000 × 512 | 4.3 |
| 50,000 × 512 | 20.9 |

## Estimated recognition latency (first detection → event)

Burst collection: 900 ms (maxFrames=10 @ 10 fps, windowMs=1000).

| Embedder/EP | + detect per frame | + embed topK=5 | total estimate | ТЗ target |
|---|---|---|---|---|
| lvface-s-glint360k/cpu | 8.1 | 53.6 | 987 ms | ≤ 1500 ms ✅ |
| lvface-s-glint360k/cuda | 8.1 | 3.4 | 937 ms | ≤ 1500 ms ✅ |
| lvface-t-glint360k/cpu | 8.1 | 21.8 | 955 ms | ≤ 1500 ms ✅ |
| lvface-t-glint360k/cuda | 8.1 | 2.1 | 935 ms | ≤ 1500 ms ✅ |
| lvface-b-glint360k/cpu | 8.1 | 113.9 | 1047 ms | ≤ 1500 ms ✅ |
| lvface-b-glint360k/cuda | 8.1 | 5.7 | 939 ms | ≤ 1500 ms ✅ |

Engine RSS after loading models: 2323 MiB.
