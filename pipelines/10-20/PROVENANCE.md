# 10-20 source provenance

This table distinguishes byte-identical executed snapshots from the two files
that were minimally sanitized for publication.

| Published path | Executed source SHA-256 | Publication relation |
|---|---|---|
| `generation/annotate_visual_seconds.py` | `8bb79ede59004d1ebb175e892183fffbd6fd21174acfdfbe7ba21212c7ba9155` | Same implementation; module docstring expanded |
| `generation/qa_gen/common.py` | `824bec5a3bda647601ff11c2c7c3323a015475552aa899e0cc1e35ab46243744` | Same logic; machine key-file read replaced by `OPENROUTER_API_KEY` plus missing-key guard |
| `generation/qa_gen/gen_v3.py` | `59066b91de575e73dc875076bdb87ed31a72b0e7b7b2c22429148a30ab45914d` | Byte-identical |
| `generation/qa_gen/gen_hr.py` | `d01924323f80426d8362b8d2a1177bbd8e2e1c7a8d207f4648e595d9dc7564b5` | Byte-identical |
| `generation/qa_gen/gen_pg.py` | `1a1062d0dfbbd985aa1862177ce9800850c334d265aee0486308612767504b54` | Byte-identical |
| `generation/qa_gen/fix_first_person.py` | `718cc30b3cebfd7b3cd9d5885d945fa52977982f8bc6bfa76fad636cc9f84d9e` | Byte-identical |
| `generation/qa_gen/fix_language.py` | `ce0207bf4153b3cb0643791ceacd01df61e34cec6602767735390ae05a24bc52` | Byte-identical |
| `generation/qa_gen/postprocess_v3b.py` | `0a0bb6df6b5978b3afb0926ac1c01fccc4b427e411f62788d671373091539519` | Byte-identical |
| `generation/qa_gen/select_pool.py` | `9d1e4f79a1e8348b06aa0129e45a154dfb9b702a1aaa1082dee7c4fa7950768d` | Byte-identical |
| `generation/qa_gen/select_final.py` | `179efdd2e9bbd49b963fb43e7b153e9de5d55ccf736f1aad820ebc25a9a4b2be` | Byte-identical |
| `prompts/qa_generation.md` | `8bf85b786924439b568dc28b988e430c3bb9d75129f983972ecacf83481a692a` | Code-generated Prompt dump; trailing Markdown spaces normalized |

`generation/config.py` is a new environment-only adapter. It replaces the
executed machine-local configuration and intentionally contains no credential,
private path, media, or task content.

The evaluation and Judge files already present in `evaluators/`,
`runners/`, and `postprocessing/` are the 10 FPS publication snapshots.
Raw run artifacts remain external and are referenced only through the
environment contract in `env.example`.
