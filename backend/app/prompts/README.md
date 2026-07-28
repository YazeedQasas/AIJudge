# Prompt versions
<!-- Used YAML for easy readability -->

Each prompt version is **one Markdown file** named `<name>_<version>.md` (e.g. `judge_v1.md`).
The file has two parts:

1. A **YAML frontmatter** block (between the `---` fences) holding the metadata.
2. The **prompt template** below the fences — the raw text with `{placeholders}`
   the loader fills in at request time.

## Metadata fields

| Field       | Meaning                                                        |
|-------------|----------------------------------------------------------------|
| `version`   | Version id, e.g. `v1`, `v2`. Must match the filename suffix.    |
| `name`      | Prompt family, e.g. `judge`. Must match the filename prefix.    |
| `author`    | Who wrote/changed this version.                                |
| `created`   | ISO date the version was created.                             |
| `model`     | The model this wording was tuned/tested against.              |
| `description` | What this version is and how it behaves.                    |
| `changelog` | List of what changed, newest last. This is the audit trail.   |
| `generation` | Sampling params (`temperature`, `top_p`, `top_k`) this version runs with. Belongs to the version — same wording behaves differently at different settings. |
| `retrieval`  | Retrieval context in play (`embedding_model`, `embedding_dimension`). Recorded for attribution; **not** controlled by the prompt. |

## Placeholders

The template must contain these, and only these, `{...}` placeholders:

- `{sources_block}` — the numbered legal sources, assembled from retrieved chunks.
- `{question}` — the user's question.

## Adding a new version

Copy the latest file to the next number (`judge_v1.md` → `judge_v2.md`), edit the
wording, and update the frontmatter (`version`, `created`, `changelog`). Never edit
an existing version's text in place — that defeats the point of versioning and breaks
rollback. To roll back, point the active version (config) at an older file.
