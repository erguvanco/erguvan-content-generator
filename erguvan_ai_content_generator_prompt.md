# 1. Mission (One‑Sentence Objective)
Design and implement a **scalable AI system** that ingests competitor sustainability‑related documents, learns their stylistic/structural patterns, and—on demand—produces **original, client‑ready content** in Erguvan’s brand voice.

# 2. Domain Context
Erguvan provides climate policy, carbon accounting, ESG management, and carbon‑tax advisory services.  
Competitor documents often include:
- Climate policy whitepapers & country guides  
- Carbon‑footprint methodology notes  
- ESG strategy decks & KPIs  
- Carbon‑pricing / tax compliance briefings  
- Net‑zero roadmap case studies  
*(Treat this list as non‑exhaustive; auto‑expand when new sub‑domains appear.)*

# 3. Development Setting
- **IDE**: Cursor  
- **Model sandbox**: Claude‑Code inside Cursor Agent Development Mode  
- **Languages**: Python (primary), with optional TypeScript for UI helpers  
- **Frameworks/Libraries**: LangChain or LlamaIndex (vector store), Pydantic, FastAPI, Docx/PDF parsers (`python-docx`, `pdfplumber`), SentenceTransformers or OpenAI embeddings  
- **Persistence**: Local SQLite or Postgres (dev) → easily swappable to cloud DB in prod  
- **Version control**: Git (use conventional commits)

# 4. Inputs
- **/samples/** (`.docx`, `.pdf`, `.md`, `.pptx`) — competitor source docs  
- **Prompt payload** via API/CLI containing:
  - `desired_length` (words or pages)  
  - `audience` (e.g., CFO, Sustainability Lead)  
  - `topic` (e.g., “Carbon Border Adjustment Mechanism in EU”)  
  - `style_override` (optional adjectives)  

# 5. Core Functional Requirements

| Module | Purpose | Key Notes |
|--------|---------|-----------|
| **Loader** | Parse & normalize docs → plain text + metadata. | Preserve headings; strip boilerplate legalese. |
| **Analyzer** | Extract style fingerprints (tone, tense, avg. sentence length, formality score, common headers). | Output structured JSON schema. |
| **Vector Index** | Store embeddings for semantic similarity, citation retrieval. | Top‑k query speed < 300 ms on 1k docs. |
| **Generator** | Compose new content conditioned on: prompt payload + nearest‑neighbor style exemplars + Erguvan brand guide. | Must produce *original*, non‑derivative text; include auto‑citations to public regs where helpful. |
| **Evaluator** | Automated tests: plagiarism score < 10 %, Flesch readability ≥ 50, brand‑tone match ≥ 90 % vs. template. | Fail generation if thresholds not met. |

# 6. Non‑Functional / Quality Constraints
- **Brand Voice**: authoritative yet approachable; avoid jargon without definition; always solutions‑oriented.  
- **Security**: no outbound calls unless whitelisted; sanitize PDF/Docx to mitigate macros.  
- **Scalability**: modular services; design for future move to serverless.  
- **Logging & Observability**: structured logs (JSON); expose `/metrics` for Prometheus.  
- **Testing**: 90 %+ unit‑test coverage; include integration test that generates a 2‑page ESG brief.

# 7. Output Contract
```json
{
  "title": "string",
  "author": "Erguvan Advisory AI",
  "created_utc": "ISO-8601",
  "audience": "string",
  "word_count": 1234,
  "sections": [
    { "heading": "string", "body": "string" }
  ]
}
```
Also write an editable `.docx` to `/generated/` using the same structure.

# 8. Acceptance Criteria (Definition of Done)
1. CLI demo: `python main.py --topic "EU CBAM" --audience CFO --desired_length 1200` outputs doc in < 60 s.  
2. Evaluation module passes all thresholds (see §5).  
3. README includes quick‑start, architecture diagram, and extensibility notes.  

# 9. Step‑By‑Step Development Plan (Agent must follow or propose improvements)
1. **Scaffold repo** with virtual env, pre‑commit hooks.  
2. Implement **Loader** & write unit tests.  
3. Implement **Analyzer**; generate JSON style‑profiles.  
4. Build **Vector Index** & similarity search.  
5. Draft **Generator** chain (RAG + prompt template).  
6. Integrate **Evaluator**; set fail‑fast rules.  
7. Wire up CLI/API endpoint.  
8. Add docs, diagrams, and run end‑to‑end demo.  
*(If you foresee a better sequence, explain and request approval before diverging.)*

# 10. Edge‑Case & Risk Reminders
- Single sample doc → fall back to generic Erguvan style guide.  
- Multi‑language docs (EN/TR) → detect language, ensure output in prompt language.  
- Highly technical tables → preserve numerical fidelity; rounding rules per ISO 14064.  
- Confidential info ↔ redact before vector‑store ingestion.  

# 11. Few‑Shot Style Prompts (for Generator) — example snippet
> **System**: “You are Erguvan’s Sustainability Writing Assistant…”  
> **User**: “Draft a 500‑word primer on scope‑3 emissions for procurement managers.”  
> **Assistant (ideal)**: “**Introduction** — Procurement decisions often hide…” (clear structure, actionable tone, 2 bullet lists, citation stubs).

# 12. Simulation Test
After coding each major module, run an internal prompt:  
> “Generate a 700‑word advisory note on Turkey’s 2025 carbon price trajectory for CEOs.”  

Verify output meets §5 & §6 automatically.

---

## End of Prompt
**Deliverable to Erguvan**: Push code to repo + attach generated demo document.
