# Kloter — Architecture du pipeline

> Transcription audio multilingue avec diarization locuteurs

## Vue d'ensemble

```
audio.mp3 → Conversion → VAD → ┌ Whisper large-v3 ┐ → Merge tokens → Wav2vec2 → Matching → JSON
                                └ Diarization      ┘       ↑ lang/segment     ↑ lang/segment
                                       └──────────────────────────────────────────┘
```

Un seul script Python. Un seul appel. Fichiers en sortie.

**Langue auto-détectée** par Whisper per segment — pas de flag `--lang` (sera réintroduit plus tard comme optimisation).

```bash
kloter audio.mp3                    # → audio.transcription.json + audio.transcription.md
kloter audio.mp3 --format json      # → audio.transcription.json only
kloter audio.mp3 --format md        # → audio.transcription.md only
kloter audio.mp3 --stdout           # → JSON on stdout (for piping / Ruby)
```

---

## Étape ① — Conversion audio

**Entrée** : n'importe quel format (mp3, ogg, m4a, flac, wav, webm…)

**Sortie** : WAV 16kHz mono float32 (tableau numpy)

**Outil** : `ffmpeg`

**Pourquoi ce format ?**

- **16kHz** : sample rate de référence pour whisper, wav2vec2, pyannote
- **mono** : la stéréo n'apporte rien, les modèles travaillent en mono
- **float32** : format natif des tenseurs PyTorch
- **WAV** : format non compressé lisible par tous les modèles

```python
def load_audio(path):
    result = subprocess.run(
        ["ffmpeg", "-i", path,
         "-f", "wav", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio[22:]  # skip WAV header (44 bytes = 22 int16 samples)
```

---

## Étape ② — VAD Pyannote (léger)

**Modèle** : `pyannote/segmentation` (~0.7s pour 27s d'audio, déjà en cache)

**Entrée** : tableau numpy 16kHz

**Sortie** : segments de parole `[{start, end}, ...]`

**Pourquoi la VAD légère et pas la diarization complète ici ?**

| | VAD seule | Diarization complète |
|---|---|---|
| Temps | ~0.7s | ~15s |
| Sortie | segments parole | segments parole + locuteurs |
| Intérêt | Donne les segments à Whisper tout de suite | Il faut attendre le clustering |

On a besoin des segments de parole **immédiatement** pour lancer Whisper et la diarization en parallèle. Attendre 15s la diarization pour découper l'audio retardait tout le pipeline.

La diarization tourne **en parallèle** de Whisper — elle aura terminé avant que Whisper ne finisse.

```python
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

segmentation_model = Model.from_pretrained("pyannote/segmentation")
vad = VoiceActivityDetection(segmentation=segmentation_model)
vad.instantiate({
    "onset": 0.500,
    "offset": 0.363,
    "min_duration_on": 0.100,
    "min_duration_off": 0.100,
})

audio_data = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}
vad_result = vad(audio_data)

speech_segments = [
    {"start": round(s.start, 3), "end": round(s.end, 3)}
    for s in vad_result.itersegments()
]
```

---

## Étape ③ — Whisper.cpp large-v3 f16 (parallèle avec ④)

**Modèle** : `ggml-large-v3.bin` (f16, non quantifié, ~3 Go)

**Entrée** : chunks WAV découpés par la VAD

**Sortie** : tokens avec timestamps + probabilité + **langue détectée par segment**

**Pourquoi whisper.cpp et pas openai-whisper (Python) ?**

| | openai-whisper (Python) | whisper.cpp (C/C++) |
|---|---|---|
| Vitesse CPU (27s audio, 8 threads) | ~90s+ | ~25-60s |
| RAM modèle | ~3 Go (float32) | ~3 Go (f16) |
| Appel depuis Python | API Python native | `subprocess.run("whisper-cli")` |
| Détection langue | Par segment, natif | `-l auto`, par chunk |
| Quantification | Non | q2→q8, contrôle fin |

**On privilégie la qualité maximale** avec le modèle non quantifié (f16, pas q5_0). La vitesse est déjà 3-4× meilleure que openai-whisper grâce au code C/C++ optimisé.

**Détection de langue automatique** : chaque segment VAD est transcrit avec `-l auto`. Whisper.cpp renvoie la langue détectée dans `result.language` du JSON de sortie.

Whisper.cpp est appelé en CLI pour chaque segment de parole. Résultat en JSON avec `-oj -ojf` (output json full).

**⚠️ Problème** : les tokens sont au niveau **sous-mot** :

```
"élect" [7.420 → 7.610]  p=0.978
"ri"    [7.610 → 7.670]  p=0.987
"ques"  [7.670 → 7.800]  p=0.999
```

→ Il faut merger ces tokens en mots complets (étape ③bis).

```bash
whisper-cli -m ggml-large-v3.bin \
  -l auto -t 8 -bo 5 -bs 5 \
  -oj -ojf -sow \
  -of /tmp/out chunk.wav
```

---

## Étape ③bis — Merge tokens → mots

**Entrée** : tokens whisper.cpp (sous-mots)

**Sortie** : mots entiers avec timestamps fusionnés

**Règle** : fusionner les tokens tant que le token suivant ne commence pas par un espace (pas de coupure de mot).

```
"élect" [7.420 → 7.610]    ┐
"ri"    [7.610 → 7.670]    ├──→ "électriques" [7.420 → 7.800]  score=moyenne
"ques"  [7.670 → 7.800]    ┘

"Saint" [3.350 → 3.640]    ──→ "Saint" [3.350 → 3.640]
"-"     [3.640 → 3.690]    ┐
"Y"     [3.690 → 3.730]    ├──→ "-Yves" [3.640 → 3.990]
"ves"   [3.730 → 3.990]    ┘
```

```python
def merge_tokens_to_words(tokens, offset=0.0):
    words = []
    current = None

    for token in tokens:
        text = token["text"]
        if text.startswith("[_") or not text:
            continue

        is_continuation = not text.startswith(" ")

        if current and is_continuation:
            current["end"] = round(token["end"] + offset, 3)
            current["word"] += text.strip()
            current["scores"].append(token["probability"])
        else:
            if current:
                current["probability"] = round(sum(current["scores"]) / len(current["scores"]), 3)
                del current["scores"]
                words.append(current)
            current = {
                "start": round(token["start"] + offset, 3),
                "end": round(token["end"] + offset, 3),
                "word": text.strip(),
                "probability": None,
                "scores": [token["probability"]],
            }

    if current:
        current["probability"] = round(sum(current["scores"]) / len(current["scores"]), 3)
        del current["scores"]
        words.append(current)

    return words
```

---

## Étape ④ — Diarization Pyannote (parallèle avec ③)

**Modèle** : `pyannote/speaker-diarization-community-1`

**Entrée** : tableau numpy complet (pas seulement les segments parole)

**Sortie** : `[{start, end, speaker}, ...]`

**Pourquoi en parallèle de Whisper ?**

Whisper large-v3 non quantifié est le goulot d'étranglement (~90s+ en CPU). La diarization prend ~15s en CPU.
En les lançant en parallèle, la diarization sera terminée bien avant la fin de Whisper.
Résultat : le temps total = max(whisper, diarization) + wav2vec2, pas la somme.

```
              0s    5s    10s   15s   20s   25s   30s   35s   40s   ...   90s
 ③ Whisper    ████████████████████████████████████████████████████████████████
 ④ Diarization████████████████
 ⑤ Wav2vec2                                                          ████
                                                                     ↑ diar déjà prête
```

**Attention** : la diarization utilise le **même modèle de segmentation** que la VAD (étape ②).
Le modèle est chargé une seule fois et partagé.

**⚠️ Modèle gated** : nécessite un token HuggingFace et l'acceptation des conditions sur :
https://huggingface.co/pyannote/speaker-diarization-community-1

```python
from pyannote.audio import Pipeline as PyannotePipeline

diar_pipeline = PyannotePipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=hf_token,
)

output = diar_pipeline(
    {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000},
    min_speakers=min_speakers,
    max_speakers=max_speakers,
)

diar_segments = [
    {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True)
]
```

---

## Étape ⑤ — Wav2vec2 Alignment multilingue

**Modèles** : sélectionnés automatiquement selon la langue détectée par Whisper **par segment**

| Langue | Modèle wav2vec2 |
|---|---|
| `fr` | `jonatasgrosman/wav2vec2-large-xlsr-53-french` |
| `en` | `jonatasgrosman/wav2vec2-large-xlsr-53-english` |
| `de` | `jonatasgrosman/wav2vec2-large-xlsr-53-german` |
| `es` | `jonatasgrosman/wav2vec2-large-xlsr-53-spanish` |
| `it` | `jonatasgrosman/wav2vec2-large-xlsr-53-italian` |
| ... | ... (whisperx lookup table) |

**Entrée** :
- mots whisper avec langue attachée (étape ③bis)
- tableau numpy audio

**Sortie** : mots avec timestamps précis (~10-30ms) + score d'alignement

**Pourquoi wav2vec2 en plus de whisper ?**

| | Whisper (cross-attention) | Wav2vec2 (CTC forced align) |
|---|---|---|
| Précision | ~100-200ms | ~10-30ms |
| Mots courts ("je", "?") | mauvais | bon |
| Respecte les silences | ✅ Oui | ❌ Étalement sur les gaps |

**Les deux se complètent** :
- Whisper détecte les gaps → la VAD découpe → wav2vec2 n'a plus de gaps à étaler
- Wav2vec2 affine les timestamps à l'intérieur des segments parole

**⚠️ Pas d'étalement** car les segments sont déjà découpés par la VAD.

### Alignement multilingue par segment

Chaque segment a sa propre langue (détectée par Whisper). On charge un modèle wav2vec2 différent par langue — avec un cache pour ne pas recharger.

```python
from whisperx.alignment import load_align_model, align

ALIGN_MODELS = {}  # cache: lang → (model, metadata)
MAX_LANGUAGES = 3   # limiter l'empreinte mémoire

def get_align_model(lang: str):
    """Charge et met en cache un modèle wav2vec2 par langue."""
    if lang not in ALIGN_MODELS:
        model, metadata = load_align_model(lang, "cpu")
        ALIGN_MODELS[lang] = (model, metadata)
    return ALIGN_MODELS[lang]

def detect_languages(segments: list[dict]) -> list[str]:
    """Top-N langues par durée totale de parole."""
    from collections import defaultdict
    duration_per_lang = defaultdict(float)
    for seg in segments:
        lang = seg["language"]
        dur = seg["end"] - seg["start"]
        duration_per_lang[lang] += dur
    sorted_langs = sorted(duration_per_lang.items(), key=lambda x: -x[1])
    return [lang for lang, _ in sorted_langs[:MAX_LANGUAGES]]

def align_words(segments: list[dict], audio: np.ndarray) -> list[dict]:
    """Aligne chaque segment avec le modèle wav2vec2 de sa langue."""
    supported_langs = detect_languages(segments)
    majority_lang = supported_langs[0]

    all_aligned_words = []
    for seg in segments:
        lang = seg["language"]
        
        # Fallback sur la langue majoritaire si pas dans le top-N
        if lang not in supported_langs:
            lang = majority_lang
            seg["language_fallback"] = True
        
        model, metadata = get_align_model(lang)
        
        wxs_segment = {
            "start": seg["start"],
            "end": seg["end"],
            "text": " ".join(w["word"] for w in seg["words"]),
        }
        
        aligned = align([wxs_segment], model, metadata, audio, "cpu")
        
        for w in aligned["words"]:
            w["language"] = lang
            if seg.get("language_fallback"):
                w["language_fallback"] = True
            all_aligned_words.append(w)
    
    return all_aligned_words
```

### Gestion mémoire

Chaque modèle wav2vec2 ~360 Mo en RAM. Avec `MAX_LANGUAGES=3` :

| # langues | RAM wav2vec2 |
|---|---|
| 1 (monolingue) | ~360 Mo |
| 2 | ~720 Mo |
| 3 (max) | ~1.1 Go |

Les segments dans une langue minoritaire (hors top-3) sont alignés avec le modèle de la langue majoritaire — qualité dégradée mais acceptable.

---

## Étape ⑥ — Matching locuteurs

**Entrée** :
- mots alignés `[{start, end, word, probability, align_score, language}, ...]`
- diarization `[{start, end, speaker}, ...]`

**Sortie** : chaque mot a un `speaker`

### Algorithme de base : intersection temporelle

```
Pour chaque mot m:
    best_speaker = "UNKNOWN"
    best_overlap = 0
    pour chaque segment diarization d:
        overlap = min(m.end, d.end) - max(m.start, d.start)
        si overlap > best_overlap:
            best_overlap = overlap
            best_speaker = d.speaker
    m.speaker = best_speaker
```

### Améliorations

1. **Filtrage mots bas score** : si `align_score < 0.1`, le mot est douteux → ne pas le compter dans le vote de segment
2. **Cohérence intra-segment** : dans une phrase courte (<2s), il y a quasi toujours 1 seul locuteur → vote majoritaire sur tous les mots du segment
3. **Chevauchements diarization** : si 2 locuteurs se chevauchent sur un mot → marquer comme `SPEAKER_00+SPEAKER_01` (overlap)
4. **Propagation** : si un mot n'a aucun chevauchement (dans un trou de diarization) → prendre le locuteur du mot précédent le plus proche

---

## Étape ⑦ — Formatting final

### Fichiers de sortie

**Convention de nommage** : même dossier, même basename, double-extension `.transcription.*`

```
audio.mp3 → audio.transcription.json     (données complètes, machine-readable)
              audio.transcription.md        (résumé lisible, human-readable)
```

**Comportement par défaut** (`kloter audio.mp3`) :
- Écrit les deux fichiers dans le même dossier que l'audio source
- Pas de sortie sur stdout (sauf `--stdout`)

**Flags** :

| Flag | Comportement |
|---|---|
| *(default)* | Écrit `.transcription.json` + `.transcription.md` dans le dossier de l'audio |
| `--format json` | Écrit uniquement `.transcription.json` |
| `--format md` | Écrit uniquement `.transcription.md` |
| `--output-dir /tmp/out` | Écrit les fichiers dans ce dossier au lieu du dossier source |
| `--stdout` | Envoie le JSON sur stdout (pour pipe / Ruby), n'écrit pas de fichiers |

### Format JSON — `.transcription.json`

Données complètes : timestamps, scores, locuteurs, langues. Tout ce que les étapes précédentes ont calculé.

```json
{
  "audio": "audio.mp3",
  "duration": 26.8,
  "languages": {
    "fr": 22.5,
    "en": 4.3
  },
  "words": [
    {
      "start": 0.379,
      "end": 0.540,
      "word": "Si",
      "probability": 0.39,
      "align_score": 0.48,
      "language": "fr",
      "speaker": "SPEAKER_00"
    },
    ...
  ],
  "segments": [
    {
      "start": 0.379,
      "end": 1.586,
      "speaker": "SPEAKER_00",
      "language": "fr",
      "text": "Si, vous faites une photo de quoi là ?"
    },
    ...
  ],
  "diarization": [
    {"start": 0.267, "end": 1.533, "speaker": "SPEAKER_00"},
    ...
  ],
  "speech_segments": [
    {"start": 0.031, "end": 1.702},
    ...
  ]
}
```

### Format Markdown — `.transcription.md`

Résumé lisible : texte groupé par locuteur, avec timestamps lisibles.

```markdown
# Transcription — audio.mp3

**Duration**: 26.8s | **Languages**: fr (22.5s), en (4.3s) | **Speakers**: 2

---

## SPEAKER_00

**[0:00–0:26]**

Si, vous faites une photo de quoi là ?

Non, pas du tout.

## SPEAKER_01

**[0:15–0:22]**

We should probably take a different approach.

---

_Generated by kloter v0.1.0_
```

### Contenu par format

| Champ | JSON | Markdown |
|---|---|---|
| Mots avec timestamps | ✅ | ❌ |
| Mots avec speakers | ✅ | ❌ |
| Mots avec scores | ✅ | ❌ |
| Mots avec langue | ✅ | ❌ |
| Segments speaker + texte | ✅ | ✅ (groupés par locuteur) |
| Segments diarization | ✅ | ❌ |
| Segments parole (VAD) | ✅ | ❌ |
| Langues + durée | ✅ | ✅ (en-tête) |
| Durée totale | ✅ | ✅ (en-tête) |
| Nombre de locuteurs | ✅ (implicite) | ✅ (en-tête) |
| Version kloter | ❌ | ✅ (pied de page) |

**JSON = le dataset complet. Markdown = le résumé lisible.**

### Pourquoi pas SRT/VTT/TXT ?

SRT et VTT sont des formats vidéo-niche — ils peuvent être générés plus tard à partir du JSON.
TXT est subsumé par le Markdown.
On garde deux formats qui couvrent les deux besoins : machine et humain.

---

## Timeline d'exécution

### Sur serveur (16 threads, 32 Go RAM)

```
              0s     5s    10s    15s    20s    25s    30s    35s    40s   ...  90s+
              │      │      │      │      │      │      │      │      │          │
 ① Conversion │███│      │      │      │      │      │      │      │          │
 ② VAD        │████│      │      │      │      │      │      │      │          │
 ③ Whisper    │      ████████████████████████████████████████████████████████████
 ④ Diarization│      ████████████│      │      │      │      │      │          │
 ⑤ Wav2vec2   │      │      │      │      │      │      │      │      │      ████│
 ⑥ Matching   │      │      │      │      │      │      │      │      │      ██│
 ⑦ Formatting │      │      │      │      │      │      │      │      │      █│
              │      │      │      │      │      │      │      │      │          │
              ├──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┤          │
               ③ + ④ en parallèle                        Total ~90s+ (CPU)
```

### Sur laptop (8 threads, 16 Go RAM)

```
 ①+②  ~1s
 ③     ~90s+ (large-v3 float32, CPU)
 ④     ~15s  (en parallèle de ③)
 ⑤     ~4s
 ⑥+⑦  <1s

Total : ~95s+ pour 27s d'audio (CPU, qualité maximale)
```

**⚠️ Le CPU est lent pour large-v3 float32.** Optimisations futures :
- `--lang fr` pour court-circuiter la détection de langue → petit gain
- Quantification (q5_0, int8) → 2-3× plus rapide, légère perte qualité
- `whisper-large-v3-turbo` → 4× plus rapide, qualité comparable
- GPU → 8-10× plus rapide

---

## Gestion des ressources CPU

Les briques ML pompent toutes les ressources. Sur un serveur avec 16 threads :

```
Whisper (Python) : torch.set_num_threads(8)  (8 threads)
Pyannote          : torch.set_num_threads(4)
Wav2vec2          : torch.set_num_threads(4)
```

Quand ③ et ④ tournent en parallèle : 8 + 4 = 12 threads → reste 4 pour l'OS.
Quand ⑤ tourne seul : 4 threads → reste 12 pour le reste.

**Un seul job à la fois** (Sidekiq concurrence=1 en production).

---

## Dépendances

### Système

- `ffmpeg` : conversion audio + extraction chunks

### Python (venv)

- `whisper-cli` : whisper.cpp (CLI natif C/C++)
- `torch` (CPU)
- `torchaudio` (CPU)
- `pyannote.audio`
- `whisperx` (pour l'alignement wav2vec2)
- `numpy`

### Modèles (cache local)

| Modèle | Taille | Source |
|---|---|---|
| `ggml-large-v3.bin` | ~3 Go (f16) | HuggingFace ggerganov/whisper.cpp |
| `pyannote/segmentation` | ~6 Mo | Bundled in speaker-diarization-community-1 |
| `pyannote/speaker-diarization-community-1` | ~30 Mo | HuggingFace (gated, token requis) |
| `wav2vec2-large-xlsr-53-{lang}` | ~360 Mo × N langues | HuggingFace (auto-download via whisperx) |

**⚠️ Un seul gated repo à accepter** : `pyannote/speaker-diarization-community-1`. Le modèle de segmentation est inclus dans le sous-dossier `segmentation/` du repo.

---

## Appel depuis Rails (plus tard)

```ruby
# Dans un Sidekiq job — JSON sur stdout, pas de fichiers
result_json = `kloter #{audio_path} --stdout`
result = JSON.parse(result_json)

# Ou : écrit les fichiers et lit le JSON
`kloter #{audio_path} --output-dir #{output_dir}`
result = JSON.parse(File.read("#{output_dir}/#{File.basename(audio_path, '.*')}.transcription.json"))
```

Pas de microservice. Pas de daemon. Un simple appel système.
