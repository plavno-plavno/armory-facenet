# FaceID Engine

Локальный офлайн-сервер распознавания лиц по ТЗ `TZ-face-recognition-engine.md`:
YuNet (детекция) + LVFace (эмбеддинги) на ONNX Runtime, файловое зашифрованное хранилище,
REST/WS API, потоковый пайплайн (веб-камера / RTSP / файл), Electron-оболочка с админкой.

## Быстрый старт (разработка)

```bash
npm ci
bash models/download.sh                 # YuNet + LVFace-T/S/B, sha256 в models/manifest.json
npm run build                           # shared + engine

# веб-режим без Electron: движок + админка в браузере на http://127.0.0.1:47810/
FACEID_PASSWORD='…' npm run web         # печатает адрес и токен API; токен вводится на странице входа
# Electron-приложение (админка + трей + захват веб-камеры)
npm start -w faceid-desktop             # на Ubuntu 24.04 в dev-режиме может понадобиться --no-sandbox
npm run dist -w faceid-desktop          # установщики: NSIS / dmg / AppImage+deb
```

API: `http://127.0.0.1:47810/api/v1`, заголовок `Authorization: Bearer <token>`,
OpenAPI — `GET /api/v1/openapi.json`.

```bash
curl -H "Authorization: Bearer $T" \
  -F 'data={"firstName":"Иван","lastName":"Петров","consent":{"obtained":true}}' \
  -F photos=@1.jpg -F photos=@2.jpg  http://127.0.0.1:47810/api/v1/persons
```

### Веб-режим

`npm run web` собирает админку и запускает движок, который раздаёт её на `/`. Вход — по токену API
(хранится только в `sessionStorage` вкладки). Веб-камеры читаются через ffmpeg (Linux: v4l2 `/dev/videoN`),
RTSP и файлы — как обычно. Ключ хранилища — из `FACEID_PASSWORD` (scrypt); тот же каталог данных
открывается и Electron-приложением при парольном режиме. По умолчанию слушается только `127.0.0.1`;
доступ из сети — только с TLS (`server.tls`, §12.3).

Регистрация в админке принимает фото из файлов и снимки с камеры: блок «Снимок с камеры» показывает живое
превью запущенного потока, кнопка «Снять с камеры» берёт `enroll.captureFrames` лучших кадров с одним лицом
(`POST /streams/:id/capture`, ничего не сохраняет); при создании они помечаются `source: "camera"` (`photoSources`).

### Видеокарта (NVIDIA)

onnxruntime-node собран под CUDA 13. При наличии драйвера NVIDIA движок (`models.executionProvider = auto`)
всегда пробует CUDA: сам находит и предзагружает CUDA 13 runtime + cuDNN 9 из системы, из `.cuda-libs/`
проекта, из `FACEID_CUDA_LIBS` или `<resources>/cuda-libs` — как бы он ни был запущен. Если не получилось,
работает на CPU, а причина видна в логе и в `/health` → `warnings` / `models.embedder.fallbackReason`.

```bash
npm run cuda:install      # cuBLAS / cudart / cuDNN (~1.5 ГБ) в .cuda-libs/ внутри проекта, без установки в систему
```

Упакованные установщики CUDA-библиотеки не содержат: на целевой машине нужен CUDA 13 runtime
(или используется CPU; для слабых CPU — `models.embedder = lvface-t-glint360k`).

## CLI

| Команда | Назначение |
|---|---|
| `npm run identify -- <img> --gallery <dir>` | M1: распознать лица на фото по галерее «папка = человек» (или `--data-dir` хранилища) |
| `npm run bench -- --ep cpu,cuda` | Бенчмарк → `docs/benchmark.md` |
| `npm run calibrate -- --dataset <dir> --camera-profile <name>` | Калибровка порогов (§15.3) → report.md, csv, profile.json; `--apply --token` применяет через `/config` |
| `npm run parity:ref` | Пересчитать Python-эталоны для parity-тестов |

## Тесты

`npx vitest run` — 214 тестов: parity Node↔Python (§15.1), юнит (декодер YuNet, NMS, Umeyama,
метрики качества, трекер, машина состояний, шифрование, kill-тест атомарной записи, индекс),
интеграционные (все эндпоинты §10 с кодами ошибок, поток из видеофайла → WS/webhooks/журнал/снимки,
рестарт и пересборка повреждённого кэша, отсутствие ПДн в логах).

Результаты parity: ошибка точек 0.008 px (порог 1), MAD выравнивания ≤ 0.19 (порог 1.0),
косинус эмбеддингов 1.000000 (порог 0.999) для LVFace-T/S/B.

## Структура

Соответствует Приложению A ТЗ: `apps/engine` (server, pipeline, vision, gallery, store, jobs, config, cli),
`apps/electron` (main, preload, capture, admin-ui), `packages/shared` (zod-схемы, типы событий, клиент API),
`models/`, `tests/parity/python` (эталоны), `tests/fixtures`, `docs/`.

## Решения и отступления от ТЗ

- **Предобработка LVFace зафиксирована** по `inference_onnx.py`: RGB, `(x−127.5)/127.5`, вход `[N,3,112,112]`, динамический batch, D=512 (манифест).
- Детектор всегда на CPU (мелкая модель с переменным размером входа); `executionProvider` относится к эмбеддеру. `auto` пробует CUDA (Linux) / DirectML (Windows) и откатывается на CPU.
- Если на входе ≤640 px лиц не найдено, при регистрации делается повторная попытка на половинном разрешении (крупные лица > 300 px, R10).
- `uncertain` публикуется как промежуточное событие (с номером попытки), итог после `maxRetries` — `unknown`.
- Во время переиндексации распознавание работает по уже переиндексированным людям (старую модель одновременно не держим), `/health` = `degraded`.
- Конфигурация потоков хранится в `streams.json.enc` (URL RTSP могут содержать пароль), в API URL отдаются с `***`.
- Токен API хранится зашифрованным DEK (`token.enc`), сам DEK — обёрнут `safeStorage` (или паролем).
- Секреты webhooks не отдаются через `GET /config` (маска `***`).
- Админка загружается со схемы `faceid://admin`; CORS в движке разрешён только для этого origin.
- Лимит регистрации настраивается: `server.enrollRatePerSec` (по умолчанию 10, §12.3).

## Риски, требующие решения заказчика

- **R1 — веса LVFace только для некоммерческого исследовательского использования.** Для коммерческой
  дистрибуции нужна лицензия правообладателя или замена модели (через манифест + `/admin/reindex`).
- **R8 — ffmpeg**: в установщик нужно положить LGPL-сборку в `vendor/ffmpeg/<platform>-<arch>/`; без неё используется ffmpeg из PATH.
- **R4 — нет liveness** (v1): система не защищена от предъявления фото/экрана. Интерфейс `LivenessChecker` готов для v2.
- Пороги `match.*` по умолчанию — заглушки; на LFW genuine ≈ 0.4–0.6, impostor ≈ 0–0.2. Рабочие значения — только после калибровки на целевой камере.
- Открытые вопросы §18 (ОС, GPU, размер галереи, целевой FAR и др.) остаются открытыми.
