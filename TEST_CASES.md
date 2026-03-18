# WordCraft - Test Cases (Manual + API)

This file contains functional and UX test cases covering:
- Tools Home (Copy/Save behavior)
- Editor modes (`Write`/`Edit`/`Rewrite`, labeled `Draft`/`Polish`/`Transform` in UI)
- NLP engine routing (selection vs blank vs sentence)
- API endpoints (`/suggest`, `/lexical`, `/oneword`, `/constraints`)
- Auth + persistence (`saved_words`, documents, session history)

---

## 1) Global Assumptions / Rules

- Blank tokens recognized: `____`, `___`, `[BLANK]`, `<blank>`, `(blank)`
- Rewrite is suppressed when a blank is present.
- Selection has higher priority than blank.
- Tools page shows `COPY` always and `SAVE` only for logged-in users.
- Tools page does not show `Insert`.
- Session history snapshots are session-only (`sessionStorage`) and update every 60s when content changes.

---

## 2) Environment + Preconditions

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- MongoDB available and writable
- Test once as guest and once as logged-in user
- Browser clipboard permissions allowed for copy tests

---

## 3) Tools Home - UI Test Cases

### TC-UI-001: Tool cards activate the correct panel
**Steps**
1. Open `/`
2. Click each card: One-Word, Smart Match, Synonyms & Antonyms, Rhyme & Homonym

**Expected**
- Active card style changes
- Panel title/help/input placeholder changes to selected tool
- No duplicate inner tool-pill navigation on home panel

### TC-UI-002: Primary interaction is clear (hero input)
**Steps**
1. Open `/`
2. Select One-Word card
3. Observe panel layout before typing

**Expected**
- A single large input is visually dominant
- CTA button text is tool-specific (`Find one word`, `Find matches`, `Get rhymes`, etc.)

### TC-UI-003: Copy button works for tools results
**Steps**
1. Generate results in any tool
2. Click `COPY` on a result
3. Paste into Notepad/Docs

**Expected**
- Exact copied word is pasted
- Button state briefly changes to `Copied` (if implemented)
- No `Insert` button present on Tools page

### TC-UI-004: Save visibility by auth state
**Steps**
1. Open Tools page as guest and generate results
2. Verify actions
3. Login, repeat

**Expected**
- Guest: only `COPY`
- Logged-in: `COPY` + `SAVE`

### TC-UI-005: Empty state behavior
**Steps**
1. Open each tool panel without submitting input

**Expected**
- Calm helper empty state appears (`Type a word to see results.`)
- No stale results from previous tool remain visible after tool switch

---

## 4) One-Word Substitution (`POST /oneword`)

### TC-OW-001: Basic phrase -> one word
**Input**
- `query`: `self obsessed`
- `context`: `neutral`

**Expected**
- Top results include close forms like `narcissist`, `egotist`, `vain` (or similar)
- Each result has reason/meaning text and score

### TC-OW-002: Multi-word concept query
**Input**
- `query`: `fear of heights`

**Expected**
- `acrophobia` appears in top results (or near top)
- Output limited to single-word candidates

### TC-OW-003: Query pattern bias - person who...
**Input**
- `query`: `a person who lies a lot`

**Expected**
- Noun-like outputs are prioritized
- Reason strings mention semantic/person-role fit

### TC-OW-004: Query pattern bias - quality of...
**Input**
- `query`: `quality of being calm under pressure`

**Expected**
- Abstract noun outputs prioritized

### TC-OW-005: Context shift effect
**Input**
- `query`: `old memory`
- compare `context=nostalgia` vs `context=horror`

**Expected**
- Top ranking shifts meaningfully by context
- Reason lines mention tone/context contribution

### TC-OW-006: Garbage/noise input handling
**Input**
- `query`: `asdfghjkl`

**Expected**
- Graceful empty or weak best-effort response
- Clear note such as `No strong matches found`

### TC-OW-007: Length guardrail
**Input**
- `limit`: `50`

**Expected**
- Response caps at service max (target <= 10)

---

## 5) Smart Match (`POST /constraints`)

### TC-SM-001: Rhyme + synonym (intersection exists)
**Input**
- `rhyme_with`: `night`
- `relation`: `synonym`
- `meaning_of`: `bright`

**Expected**
- Results rhyme with `night` and semantically align with `bright`
- Reason references both rhyme + semantic relation

### TC-SM-002: Empty intersection -> best-effort fallback
**Input**
- `rhyme_with`: `orange`
- `relation`: `synonym`
- `meaning_of`: `happy`

**Expected**
- Returns best-effort ranked items
- Includes explanatory note (`No exact matches...` / `best-effort...`)

### TC-SM-003: Antonym direction
**Input**
- `rhyme_with`: `cold`
- `relation`: `antonym`
- `meaning_of`: `warm`

**Expected**
- Tries antonym direction first, then best-effort fallback
- Reason string is explicit about constraint tradeoff

### TC-SM-004: Missing field validation
**Input**
- Missing `meaning_of` or `rhyme_with`

**Expected**
- API returns `400/422` with useful message
- UI surfaces friendly validation note

---

## 6) Lexical Tools (`POST /lexical`)

### TC-LX-001: Synonyms
**Input**
- `word`: `happy`
- `task`: `synonyms`

**Expected**
- Typical synonyms (`cheerful`, `joyful`, `glad`, etc.)
- Ranking favors common usable words

### TC-LX-002: Antonyms
**Input**
- `word`: `happy`
- `task`: `antonyms`

**Expected**
- Typical antonyms (`sad`, `unhappy`, etc.)

### TC-LX-003: Rhymes
**Input**
- `word`: `light`
- `task`: `rhymes`

**Expected**
- Relevant rhymes (`night`, `bright`, `sight`, etc.)

### TC-LX-004: Homonyms / homophones
**Input**
- `word`: `flower`
- `task`: `homonyms`

**Expected**
- `flour` appears (or equivalent homophone)
- If sparse, response includes clear note

### TC-LX-005: Explainability contract
**Input**
- Any lexical request

**Expected**
- `details[]` includes `word`, `score`, `reason` for at least returned top items

---

## 7) Editor Modes + NLP Routing (`POST /suggest`)

### TC-ED-001: Blank fill suggestions are POS-aware
**Input**
- Sentence: `The clouds shone ____ today.`
- Context: `nostalgia`
- Mode: `write`

**Expected**
- Grammar-fitting candidates (often `ADV`/`ADJ`, depending slot parse)
- Suggestions include `pos`, `score`, `note`

### TC-ED-002: Rewrite suppressed when blank exists
**Input**
- Sentence: `I like ____ to be a better.`
- Mode: `rewrite`
- Trigger: `button`

**Expected**
- Rewrite output omitted/suppressed
- UX indicates blank must be filled first

### TC-ED-003: Selection overrides blank
**Input**
- Sentence: `The afternoon light filtered through the dusty panes.`
- Selection: `dusty`
- Context: `mysterious`

**Expected**
- Suggestions target selected token replacement
- Engine does not switch to blank-fill behavior

### TC-ED-004: Write mode behavior
**Input**
- `He ran fast`
- Mode: `write`

**Expected**
- Suggestion-first behavior
- No unsolicited strong rewrite drift

### TC-ED-005: Edit mode behavior
**Input**
- `The writing was very good and made people feel things deeply.`
- Mode: `edit`

**Expected**
- Stronger polish on clarity/grammar/repetition
- Meaning preserved

### TC-ED-006: Rewrite mode is explicit trigger-only
**Steps**
1. Enter complete sentence in editor
2. Wait for auto suggestion cycle
3. Click rewrite trigger

**Expected**
- Rewrite variants generated only after explicit trigger
- No rewrite bursts while typing

### TC-ED-007: Tone differential sanity
**Input**
- Same sentence under `nostalgia` and `horror`

**Expected**
- Top 3 suggestions are materially different between contexts

---

## 8) Auth + Persistence

### TC-AUTH-001: Register and login
**Steps**
1. Register user
2. Login

**Expected**
- JWT issued and stored
- Protected endpoints accessible

### TC-DOC-001: Save document
**Steps**
1. Login
2. Write content in editor
3. Click save

**Expected**
- Document persists in DB
- Document appears in My Docs list

### TC-FAV-001: Save word from tools
**Steps**
1. Login
2. Save a word from One-Word results

**Expected**
- Row appears in `saved_words`
- Metadata includes `source=oneword`, `type=oneword`

### TC-FAV-002: Save word from lexical/smart match
**Steps**
1. Login
2. Save from Synonyms/Smart Match

**Expected**
- Entry saved with correct source/type and related fields

---

## 9) Session-only Autosave History

### TC-HIS-001: Snapshot cadence
**Steps**
1. Type in editor
2. Wait >= 60 seconds
3. Open History panel

**Expected**
- New snapshot appears only if content changed

### TC-HIS-002: Session clear behavior
**Steps**
1. Create history snapshots
2. Close browser/tab session
3. Reopen app

**Expected**
- Session history cleared

---

## 10) Error Handling + Resilience

### TC-ERR-001: Backend unreachable
**Steps**
1. Stop backend
2. Trigger any tool/search request

**Expected**
- UI shows actionable error (`Backend unreachable` or equivalent)
- No fake/mock results shown

### TC-ERR-002: Invalid payload
**Input**
- `POST /oneword` with missing/empty `query`

**Expected**
- `400/422` with clear validation message
- Frontend displays error state cleanly

### TC-ERR-003: Clipboard denied
**Steps**
1. Block clipboard permission
2. Click `COPY`

**Expected**
- Fallback copy path works (or clear failure toast)
- App remains stable

---

## 11) API Contract Spot Checks (cURL-friendly)

### TC-API-001: `/suggest` shape
Expected keys: `suggestions[]`, optional `rewrite/rewrites`, `explanation`, `detected_blank`

### TC-API-002: `/lexical` shape
Expected keys: `results[]`, `details[]` where detail has `word`, `score`, `reason`

### TC-API-003: `/constraints` shape
Expected keys: `results[]`, optional `notes`; each result has `word`, `score`, `reason`

### TC-API-004: `/oneword` shape
Expected keys: `query`, `results[]`, `note`; each result has `word`, `score`, `reason` (+ optional `meaning`)

---

## 12) Regression Locks (Must Stay Stable)

These should remain stable across releases:

- `R-01`: Blank ADV slot keeps grammar-safe adverb candidates in top results.
- `R-02`: Selection ADJ test keeps adjective-targeted replacement priority.
- `R-03`: One-Word self-focused query keeps core outputs (`narcissist`/`egotist`/`egocentric`) in top set.
- `R-04`: Smart Match hard query returns best-effort note when strict overlap fails.

Automated mapping:
- `backend/tests/test_nlp_runtime_checks.py`
- `backend/tests/test_nlp_regression_cases.py`

Run:

```bash
python -m pytest backend/tests
```

---

## 13) Filled Targeted Eval Sheet (Current Engine Snapshot)

This section captures a concrete March 18, 2026 snapshot of the current service-layer behavior for 20 representative prompts.

Notes:
- Snapshot was generated from the local backend service functions in the repo.
- Because model-network access was blocked during capture, the run used the built-in offline embedding fallback path.
- This makes the sheet useful for failure targeting, but exact rankings may shift slightly on a fully warmed local runtime with the dense embedding model loaded.

==================================================
ONE-WORD SUBSTITUTION
==================================================

1. Feature: One-Word  
Input: `fear of crowds`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `enochlophobia`, `revere`, `venerate`, `esteem`, `prise`  
Expected: `enochlophobia`, `ochlophobia`, `agoraphobia`  
Why wrong: Top-1 is strong, but the tail collapses into unrelated reverence/love words instead of staying inside the phobia family.

2. Feature: One-Word  
Input: `fear of open spaces`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `agoraphobia`, `revere`, `venerate`, `clear`, `expanse`  
Expected: `agoraphobia`  
Why wrong: The lead result is correct, but the fallback pool leaks unrelated semantic neighbors instead of remaining tightly diagnostic.

3. Feature: One-Word  
Input: `love of knowledge`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `mate`, `passion`, `beloved`, `object`, `noesis`  
Expected: `philomathy`, `epistemophilia`, `sapiophilia`  
Why wrong: It misses the technical single-word target and drifts into generic affection words plus one abstract cognition term.

4. Feature: One-Word  
Input: `hatred of mankind`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `misanthropy`, `egalitarianism`, `agape`, `lucifer`, `genie`  
Expected: `misanthropy`, `misanthropism`  
Why wrong: Top-1 is correct, but the remaining list becomes noisy and semantically unstable instead of staying close to hatred/aversion vocabulary.

5. Feature: One-Word  
Input: `a person who talks too much`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `sweater`, `hater`, `singer`, `vocalist`, `buyer`  
Expected: `chatterbox`, `blabbermouth`, `windbag`, `motor-mouth`, `gossip`  
Why wrong: Strong failure. The system is overfitting to person-like suffixes instead of actual communicative behavior.

6. Feature: One-Word  
Input: `the quality of being stubborn`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `wickedness`, `individuality`, `truthfulness`, `superiority`, `sanctity`  
Expected: `obstinacy`, `obduracy`, `stubbornness`, `intransigence`  
Why wrong: Strong failure. It recognizes the abstract-noun pattern but not the actual meaning.

==================================================
SMART MATCH
==================================================

7. Feature: Smart Match  
Input: `rhyme_with=night`, `relation=synonym`, `meaning_target=bright`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `bright`, `light`  
Expected: `bright`, `light`  
Why wrong: No major issue. This is one of the cleaner current Smart Match cases.

8. Feature: Smart Match  
Input: `rhyme_with=cold`, `relation=antonym`, `meaning_target=warm`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `bold`, `withhold`, `mold`  
Expected: `cold` as the clear exact match, or a very explicit “no exact alternative beyond cold” fallback note  
Why wrong: It misses the exact antonym-rhyme anchor and falls into rhyme-only noise.

9. Feature: Smart Match  
Input: `rhyme_with=fire`, `relation=synonym`, `meaning_target=desire`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `desire`, `require`  
Expected: `desire`, then perhaps `aspire`/`admire` only if relation logic justifies them  
Why wrong: Usable top-1, but the tail still prefers rhyme convenience over clean semantic alignment.

10. Feature: Smart Match  
Input: `rhyme_with=grief`, `relation=synonym`, `meaning_target=belief`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `belief`  
Expected: `belief`  
Why wrong: No major issue. Sparse, but correct.

11. Feature: Smart Match  
Input: `rhyme_with=stone`, `relation=antonym`, `meaning_target=soft`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `bone`, `shone`, `postpone`  
Expected: A strong fallback note with either no results or much tighter meaning-preserving best effort  
Why wrong: The current output feels rhyme-led rather than relation-led.

==================================================
LEXICAL TOOLS
==================================================

12. Feature: Synonyms  
Input: `sad`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `deplorable`, `unhappy`, `distressing`, `sorry`, `weeping`  
Expected: `sorrowful`, `melancholy`, `dejected`, `unhappy`, `mournful`  
Why wrong: It mixes multiple senses of “sad”; `deplorable` is a valid sense but not the most useful writer-facing top result here.

13. Feature: Antonyms  
Input: `warm`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `improve`, `borrow`, `love`, `shed`, `cool`  
Expected: `cold`, `cool`, `chilly`  
Why wrong: Strong failure from WordNet sense drift. Verb/object antonyms are outranking the temperature sense.

14. Feature: Rhymes  
Input: `light`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `alright`, `bright`, `alight`, `bite`, `blight`  
Expected: `bright`, `night`, `sight`, `slight`, `blight`  
Why wrong: `alright` is not the best lead for a polished writing tool, especially in formal/advanced mode.

15. Feature: Homonyms  
Input: `flower`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `flour`  
Expected: `flour`  
Why wrong: No issue. This is a clean correct case.

==================================================
EDITOR DRAFT / POLISH / TRANSFORM
==================================================

16. Feature: Editor Draft  
Input: `The room felt ____ and silent.`  
Context: `mysterious`  
Vocabulary: `advanced`  
Got: `silent`, `elusive`, `enigmatic`, `cryptic`, `veiled`  
Expected: `eerie`, `hushed`, `still`, `ominous`, `uncanny`  
Why wrong: `silent` duplicates existing context and several candidates are conceptually abstract instead of atmosphere-setting room adjectives.

17. Feature: Editor Draft  
Input: `He walked ____ into the chapel.`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `professionally`, `officially`, `ceremonially`, `courteously`, `measuredly`  
Expected: `solemnly`, `reverently`, `quietly`, `gravely`, `measuredly`  
Why wrong: It recognizes the adverb slot, but confuses institutional formality with chapel-appropriate reverence.

18. Feature: Editor Polish  
Input: `Her words were very good and made everyone feel things deeply.`  
Context: `formal`  
Vocabulary: `advanced`  
Got: `falsity`, `proclamation`, `announcement`, `illumination`, `parlance`  
Expected: A stronger rewrite and/or words like `moving`, `poignant`, `affecting`, `resonant`  
Why wrong: Strong failure. Suggestions are wrong POS and unrelated to the communicative effect being edited.

19. Feature: Editor Polish  
Input: `The memory stayed with me in a strange way.`  
Context: `nostalgia`  
Vocabulary: `advanced`  
Got: `strange`, `expound`, `memory`, `advantage`, `last`  
Expected: `hauntingly`, `oddly`, `uncannily`, or a rewrite such as “The memory lingered with me in a haunting way.”  
Why wrong: Strong failure. It surfaces repeats and unrelated lemmas instead of nostalgic/reflective refinements.

20. Feature: Editor Transform  
Input: `The letter carried a sad tone through the room.`  
Context: `melancholic`  
Vocabulary: `advanced`  
Got: Suggestions: `sad`, `improvement`, `cipher`, `bow`, `organiser`  
Expected: Suggestions like `mournful`, `somber`, `elegiac`, plus a richer rewrite  
Why wrong: The generated rewrite is serviceable, but the supporting lexical suggestions are still largely irrelevant.
