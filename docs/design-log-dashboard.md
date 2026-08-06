# Design log — dashboard redesign (2026-08-05)

## Brief (Design Studio lightweight pass)
- **Product**: internal monitoring dashboard for the evolution engine; single
  primary user (owner), checked from desktop/phone on the home LAN.
- **Evidence** (real user statements from the engagement, not personas):
  "this dash is terrible"; "I see the red and green dots appearing in the same
  spot everyday"; "we can't tell if the engine is getting smarter"; "otherwise
  we will lose track"; wants family cards with drill-down to a log.
- **Problem statement**: The owner needs the dashboard to answer, at a glance
  and in plain language, three questions in priority order — (1) is it running,
  (2) is it getting smarter, (3) is it exploring new ideas — because the old
  page buried those answers in dense skeptical prose and unexplained charts
  (gulf of evaluation; scan-hostile; no progressive disclosure).
- **Success criteria**: each question answerable in <5 seconds without reading
  a paragraph; drill-down preserved for every claim; nothing dishonest
  (no invented certainty).

## Decisions
1. **Answer-first verdict strip** — three hero cards (RUNNING / SMARTER /
   EXPLORING) with big colored verdict words and one-sentence evidence, each
   linking to its detail (canon: visible status, recognition over recall).
2. **Hierarchy reorder** — learning section (Charts C/D) directly under the
   verdicts; new 24h exploration summary card linking /explore; operational
   detail after; reference material last.
3. **Progressive disclosure** — "In one minute" map, win-criteria table, and
   glossary collapsed into <details> at the bottom (canon: progressive
   disclosure, no competing noise). Content preserved, not deleted.
4. **Kept**: the skeptical/honest tone, all existing tested components
   (charts, learning/vintage blocks, /explore lab), auto-refresh, LAN-only.
5. **Merged, not overwritten**: production's local chart improvements (axis
   labels, full retained history) synced to origin before redesign.

## Heuristic self-evaluation (top findings, fixed in this pass)
- Visible status: was buried → verdict strip (sev 3 → fixed).
- Speak user's language: "fitness/OOS/funnel" now paired with plain phrases in
  verdict sentences; jargon remains inside detail sections by design (sev 2 →
  mitigated; glossary retained).
- Recognition over recall: chart how-to-read notes retained inline (ok).
- No competing noise: 3 always-open reference cards collapsed (sev 2 → fixed).
- Aesthetic-usability: hero cards + consistent badges (polish, honest).

## Evidence status
All findings from real user statements (validated). No simulated panel used —
single-user internal tool; real-user feedback loop is direct.

## Open items
- Real-use validation: owner uses the page for a week; adjust verdict
  thresholds (e.g. EXPLORING "HEALTHY" cutoffs) to taste.
- Possible later: sparkline in the SMARTER hero once ledger cohorts >= 4.
