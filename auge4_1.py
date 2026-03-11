#!/usr/bin/env python
# -*- coding: utf-8 -*-



from pathlib import Path
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ========== Paths ==========
INPUT_PATH = Path(
    "/hits/basement/nlp/qianz/sts_embedding/dataset/original_dataset_jsonl/train.jsonl"
)
OUTPUT_PATH = Path(
    "/hits/basement/nlp/qianz/sts_embedding/dataset/augmented_dataset/t4_1.jsonl"
)

# ✅ Replace with your local model path
MODEL_PATH = "/hits/basement/nlp/qianz/models/Qwen2.5-32B-Instruct"

# ========== Generation Params ==========
BATCH_SIZE = 4
MAX_NEW_TOKENS = 1200

TEMPERATURE = 0.8
TOP_P = 0.9

# Per category: generate K candidates (Scheme A)
K_CANDIDATES = 3

# Only process [700, 1400) (0-based)
START_IDX = 0
END_IDX = 300


# ========== Prompt Templates (English) ==========

PROMPT_NEG_THEME = r"""
You are a narrative writer. Your goal is to generate the MOST USEFUL hard negative for training a text-similarity / embedding model.

TASK: Generate ONE new story of type NEG_THEME.

Definition of NEG_THEME (critical):
- The new story should be similar to the ANCHOR ONLY in THEME / TOPIC / GENRE SHELL (the broad scenario type, tone, and external conflict category).
- The new story must be clearly different in BOTH:
  (1) STRUCTURE: the major turning points / events chain
  (2) OUTCOME: the ending result / resolution template

STYLE REQUIREMENTS
- Language: English only
- Length: roughly similar to the ANCHOR
- Do NOT include lists, headings, or analysis in the output.

ANTI-COPY RULES (strict)
- Do NOT reuse any sentence from the ANCHOR.
- Do NOT reuse distinctive phrases or signature details from the ANCHOR.
- Change ALL proper names (people/places/organizations).
- Do NOT reuse the same events chain structure pattern and the main plots as the ANCHOR, even with new names/locations.

INTERNAL REASONING (DO NOT OUTPUT)
1) Read the ANCHOR and infer:
   - THEME/GENRE SHELL (genre, tone, external conflict category)
   - 4–6 turning points (abstract functions, not copied details)
   - OUTCOME TEMPLATE (abstract meaning of the ending)

   Build a STORY FINGERPRINT with 8 slots extracted from the ANCHOR:
   F1 protagonist role/archetype
   F2 primary setting type
   F3 inciting incident type
   F4 antagonist force type
   F5 central goal type
   F6 mid-story complication type
   F7 climax event type
   F8 ending scene type

2) Design a NEW story from scratch:
   - Keep only the THEME/GENRE SHELL.
   - Replace the turning-point chain so it does NOT mirror the ANCHOR.
   - Choose an ending with a clearly different OUTCOME TEMPLATE.
   - Change at least 5 of F1–F8.

3) Self-check silently:
   - Theme shell similar? YES.
   - Turning-point function overlap <= 1? YES.
   - Outcome differs in >= 2 axes? YES.
   - Fingerprint differences >= 6 of 8? YES.
   - Changed at least 3 of the 4 aspects? YES.
   - If a one-sentence logline could be written by only swapping names/places from the ANCHOR, rewrite. YES.
   - No copied phrases/names? YES.

4) If any check fails, rewrite silently until all pass.

EXAMPLES:

ANCHOR:
The foursome (Gérard Rinaldi, Jean Sarrus, Gérard Filipelli, Jean-Guy Fechner) are on a holiday. The Little Olympic flame is to be passed through their village. A grocer (Paul Préboist) calls upon them for help in decorating the village. On their job Gérard falls for the grocer's daughter Délice (Martine Kelly). However she runs away with the sportsman with the flame. The four then enter the Little Olympics to try to win her back and cause havoc in the process.
OUTPUT:
{{
"neg_theme": "The foursome head out for a quiet holiday, expecting nothing but sun and cheap food. Instead, they stumble upon a closed amusement pier where a property auction is happening under the table. A nervous shop owner begs them not to “decorate,” but to distract inspectors while he retrieves documents hidden in a freezer. One of the four befriends the owner’s daughter, only to learn she’s orchestrating the scam to leave town with the auctioneer. The quartet tries to expose the deal, bungling surveillance, misreading codes, and accidentally setting off the pier’s old rides. By dawn, the police arrive for the wrong reasons, and the four flee, empty-handed and embarrassed."}}

ANCHOR:
Anna (Gry Bay) is a single woman who seeks to maintain an active sex life while staying clear of emotional involvement, after having been jilted by the love of her life, Johan (Mark Stevens). She has a relationship with Frank (Thomas Raft), but refuses to let him move in with her. When Johan shows up again after five years absence, she starts wondering how much longer she can maintain her emotional independence, and if that is what she wants. She has sex with him, loses his telephone number and cannot contact him. She ends her affair with Frank and when she is offered a job as costume designer in a French theatre, she decides to move to Paris. She leaves her flat to her flatmate Camilla (Eileen Daly) who asks her permission to rent out the now empty room to a friend of hers. This friend turns out to be Johan, and Anna meets him as she leaves for Paris, where the local stage actors Pierre (Morten Schelbech) and Sophie (Ovidie) offer new amorous temptations, but she worries about Johan finding a new love. In the end she returns to Copenhagen and, after mistakenly thinking that Johan has been unfaithful to her, she faces her fears of commitment and is reunited with him.
OUTPUT:
{{
"neg_theme": "After a brutal breakup, Signe decides she will never again let affection turn into obligation, and she builds a life of bright nights and clean exits—lovers who don’t know her routines, weekends that don’t belong to anyone, and rules that feel like armor. The system holds until a sudden medical scare in a waiting room forces her to write down an emergency contact, and she realizes she has nobody she would actually trust to show up. On impulse she calls Elias, the one person she once loved deeply, not to rekindle anything but to ask for a practical favor: be the name on the form and nothing more. Elias agrees with a gentleness that unsettles her, and Signe starts testing her own boundaries, trying to treat intimacy like a transaction while resenting the emptiness it leaves behind. She joins a small peer group for people who fear commitment, expecting to feel superior, but instead she finds herself exposed by strangers who can describe her defenses better than she can. When Elias invites her to a quiet weekend retreat to “talk,” she goes convinced she can control the narrative—only to learn he is there to introduce the partner he is moving in with, and he wanted closure, not reunion. Signe doesn’t beg, doesn’t compete, and doesn’t run after him; she walks away furious at herself for imagining a romantic second chance. In the end she chooses neither of her old scripts—neither clinging nor fleeing into casual chaos—and begins rebuilding her life around friendships and honesty, still alone, but no longer pretending that solitude is the same thing as freedom."
}}

INPUT:
ANCHOR:
{anchor_text}

OUTPUT FORMAT (STRICT):
Return a single JSON object and nothing else:

{{
  "neg_theme": "<English story summary>"
}}
""".strip()

PROMPT_NEG_STRUCTURE = r"""
You are a narrative writer. Your goal is to produce the MOST TRAINING-USEFUL hard negative for a text-similarity / embedding model.

TASK: Generate ONE new story of type NEG_STRUCTURE.

Definition of NEG_STRUCTURE (critical):
- The new story should be similar to the ANCHOR ONLY in STRUCTURE:
  specifically, the MAJOR TURNING-POINT FUNCTIONS and their ORDER.
- The new story must be clearly different in BOTH:
  (1) THEME / GENRE SHELL (broad scenario type, tone, external conflict category)
  (2) OUTCOME TEMPLATE (the meaning of the ending result)

STYLE REQUIREMENTS
- Language: English only
- Length: roughly similar to the ANCHOR 
- Do NOT include lists, headings, or analysis in the output.

ANTI-COPY RULES (strict)
- Do NOT reuse any sentence from the ANCHOR.
- Do NOT reuse distinctive phrases or signature details from the ANCHOR.
- Change ALL proper names (people/places/organizations).
- The new theme shell must switch conflict category.A mere location swap is invalid

STRUCTURE SIMILARITY (the only allowed similarity)
- Internally extract a 5-step STRUCTURE SKELETON from the ANCHOR as TURNING-POINT FUNCTIONS:
  S1 = setup/status quo
  S2 = inciting incident forces action
  S3 = complications/escalation
  S4 = major reversal/revelation/betrayal
  S5 = climax decision + resolution
- Your new story MUST preserve S1→S5 order and keep the similar FUNCTIONAL ROLE at each step.
- However, you MUST change the concrete content of each step:
  new setting, new roles, new institutions, new objects, new scene types.

THEME + OUTCOME DIVERGENCE (strict)
- Theme/genre shell must be clearly different from the ANCHOR’s broad scenario type.
- Outcome template must differ in at least TWO axes.

INTERNAL REASONING (DO NOT OUTPUT)
1) Read the ANCHOR and infer:
   - Theme/genre shell (what kind of story it is)
   - 5-step turning-point skeleton S1–S5 (functions only)
   - Outcome template (abstract meaning of the ending)

   Build a STORY FINGERPRINT with 8 slots extracted from the ANCHOR:
   F1 protagonist role/archetype
   F2 primary setting type
   F3 inciting incident type
   F4 antagonist force type
   F5 central goal type
   F6 mid-story complication type
   F7 climax event type
   F8 ending scene type

2) Invent a NEW theme/genre shell that is clearly different.

3) Map fresh content onto S1–S5:
   - Keep the same turning-point functions in the same order,
   - Ensure concrete events are not paraphrases of the ANCHOR.
   - Change at least 7 of F1–F8.
   - Additionally, F3 (inciting incident type) and F7 (climax event type) must BOTH differ.

4) Choose an ending with a different outcome template (>= 2 axes).

5) Self-check silently:
   - S1–S5 functions match the ANCHOR’s structure? YES.
   - Theme shell clearly different? YES.
   - Outcome differs in >= 2 axes? YES.
   - Fingerprint differences >= 7 of 8? YES.
   - F3 and F7 both differ? YES.
   - Changed at least 3 of the 4 aspects? YES.
   - If a one-sentence logline could be written by only swapping names/places from the ANCHOR, rewrite. YES.
   - No copied phrases/names? YES.

6) If any check fails, rewrite silently until all pass.

EXAMPLES:

ANCHOR:
The foursome (Gérard Rinaldi, Jean Sarrus, Gérard Filipelli, Jean-Guy Fechner) are on a holiday. The Little Olympic flame is to be passed through their village. A grocer (Paul Préboist) calls upon them for help in decorating the village. On their job Gérard falls for the grocer's daughter Délice (Martine Kelly). However she runs away with the sportsman with the flame. The four then enter the Little Olympics to try to win her back and cause havoc in the process.
OUTPUT:
{{
  "neg_structure": "The foursome (a holidaying quartet) arrive just as a “Little Olympic Torch” is due to pass through their village. A grocer pleads with them to help decorate the streets, and during the frantic preparations, one of them falls hard for the grocer’s daughter, Délice. When she suddenly runs off with the confident athlete carrying the flame, the four impulsively enter the Little Olympics to win her back, causing chaos with botched routines and accidental sabotage. But as the games spiral, they discover Délice planned her escape to avoid being trapped in the village. Instead of “winning,” the foursome help her reach the train station safely, take the blame for the havoc, and watch the torch pass without her—older, wiser, and oddly relieved.
  "}}

ANCHOR:
Anna (Gry Bay) is a single woman who seeks to maintain an active sex life while staying clear of emotional involvement, after having been jilted by the love of her life, Johan (Mark Stevens). She has a relationship with Frank (Thomas Raft), but refuses to let him move in with her. When Johan shows up again after five years absence, she starts wondering how much longer she can maintain her emotional independence, and if that is what she wants. She has sex with him, loses his telephone number and cannot contact him. She ends her affair with Frank and when she is offered a job as costume designer in a French theatre, she decides to move to Paris. She leaves her flat to her flatmate Camilla (Eileen Daly) who asks her permission to rent out the now empty room to a friend of hers. This friend turns out to be Johan, and Anna meets him as she leaves for Paris, where the local stage actors Pierre (Morten Schelbech) and Sophie (Ovidie) offer new amorous temptations, but she worries about Johan finding a new love. In the end she returns to Copenhagen and, after mistakenly thinking that Johan has been unfaithful to her, she faces her fears of commitment and is reunited with him.
OUTPUT:
{{
  "neg_structure": "Anna (Gry Bay) is a single woman determined to keep her love life active while avoiding emotional dependence after being abandoned by Johan (Mark Stevens). She has an ongoing relationship with Frank (Thomas Raft) but refuses to let him move in, insisting on strict boundaries. Five years later, Johan suddenly reappears and hints that his disappearance was tied to something dangerous, though he refuses to explain. Anna is shaken, sleeps with him, and then loses his number in a chaotic night; when she tries to reach him, every trail goes cold. Feeling exposed and angry at herself, she ends things with Frank and takes a prestigious offer as costume designer at a French theatre, deciding Paris will be her clean reset. She leaves her flat to her flatmate Camilla (Eileen Daly), who asks permission to rent the empty room to a friend. That “friend” turns out to be Johan, living under a different name and using the flat as a hiding place. Anna runs into him as she is leaving for Paris, but he refuses to come with her and warns her not to return. In Paris, local performers Pierre (Morten Schelbech) and Sophie (Ovidie) offer flirtation and temptation, yet Anna can’t stop imagining Johan finding someone else—or being caught. She returns to Copenhagen to confront him, only to learn the truth: Johan has been acting as bait to draw out the people pursuing him, and Anna’s earlier contact with him put Camilla in danger. Instead of reuniting, Anna hands over Johan’s documents to the authorities, cuts ties, and disappears back to France under a new contract—choosing survival and accountability over romance."}}

INPUT:
ANCHOR:
{anchor_text}

OUTPUT FORMAT (STRICT):
Return a single JSON object and nothing else:

{{
  "neg_structure": "<English story summary>"
}}
""".strip()

PROMPT_NEG_OUTCOME = r"""
You are a narrative writer.
Write ONE hard-negative synopsis for contrastive embedding training, following the specified negative type.

TASK: Generate ONE new story of type NEG_OUTCOME.

Definition of NEG_OUTCOME (critical):
- The new story should be similar to the ANCHOR ONLY in OUTCOME TEMPLATE:
  the abstract meaning of the ending
- The new story must be clearly different in BOTH:
  (1) THEME / GENRE SHELL (broad scenario type, tone, external conflict category)
  (2) STRUCTURE (major turning points / event chain)

STYLE REQUIREMENTS
- Language: English only
- Length: roughly similar to the ANCHOR
- Do NOT include lists, headings, or analysis in the output.

ANTI-COPY RULES (strict)
- Do NOT reuse any sentence from the ANCHOR.
- Do NOT reuse distinctive phrases or signature details from the ANCHOR.
- Change ALL proper names (people/places/organizations).

OUTCOME SIMILARITY (the only allowed similarity)
- Internally summarize the ANCHOR’s OUTCOME TEMPLATE in ONE abstract sentence.
- Your new story MUST end with the same OUTCOME TEMPLATE meaning.
- BUT you must NOT reuse the same ending scene/mechanism/setting as the ANCHOR.

STRUCTURE + THEME DIVERGENCE (strict)
- Theme/genre shell must be clearly different from the ANCHOR’s broad scenario type.
  (Switch conflict category/genre, not just location.)
- Structure must be substantially different:
  - Extract 4–6 major turning points from the ANCHOR.
  - Your new story may overlap with the ANCHOR by FUNCTION at most ONCE.
  - Use a different escalation pattern and a different type of climax event.

INTERNAL REASONING (DO NOT OUTPUT)
1) Read the ANCHOR and infer:
   - Theme/genre shell
   - 4–6 turning points (functions only)
   - Outcome template (one-sentence abstract meaning)

   Build a STORY FINGERPRINT with 8 slots extracted from the ANCHOR:
   F1 protagonist role/archetype
   F2 primary setting type
   F3 inciting incident type
   F4 antagonist force type
   F5 central goal type
   F6 mid-story complication type
   F7 climax event type
   F8 ending scene type

2) Invent a NEW theme/genre shell that is clearly different.

3) Build a NEW turning-point chain (different functions; overlap <= 1).

4) Land the ending on the same OUTCOME TEMPLATE meaning:
   - Do not reuse the same ending scene/mechanism/setting as the ANCHOR.
   - Ensure F7 (climax event type) and F8 (ending scene type) BOTH differ.
   - Change at least 7 of F1–F8 overall.

5) Self-check silently:
   - Outcome template meaning matches? YES.
   - Theme shell clearly different? YES.
   - Turning-point function overlap <= 1? YES.
   - Fingerprint differences >= 7 of 8? YES.
   - F7 and F8 both differ? YES.
   - Changed at least 3 of the 4 aspects? YES.
   - If a one-sentence logline could be written by only swapping names/places from the ANCHOR, rewrite. YES.
   - No copied phrases/names? YES.

6) If any check fails, rewrite silently until all pass.

EXAMPLES:

ANCHOR:
The foursome (Gérard Rinaldi, Jean Sarrus, Gérard Filipelli, Jean-Guy Fechner) are on a holiday. The Little Olympic flame is to be passed through their village. A grocer (Paul Préboist) calls upon them for help in decorating the village. On their job Gérard falls for the grocer's daughter Délice (Martine Kelly). However she runs away with the sportsman with the flame. The four then enter the Little Olympics to try to win her back and cause havoc in the process.
OUTPUT:
{{
  "neg_outcome": "At a sprawling tech campus, four underemployed interns spend their summer doing nothing more ambitious than free lunches and office pranks until the company announces a flashy “Innovation Relay,” a public competition where teams race between demo stations carrying a prototype core meant to symbolize the firm’s future. When their manager drafts them to help set up the event, one intern, Milo, becomes infatuated with Zara, a contractor who seems bored by corporate life. The night before the relay, Zara abruptly joins a rival team led by a charismatic star engineer and disappears into restricted labs, leaving Milo convinced she has been lured away. Instead of chasing her directly, the interns decide to enter the relay under a ridiculous team name, using the competition’s checkpoints to get closer to Zara and win her attention back. Their half-baked hacks and improvised gadgets backfire spectacularly, triggering sprinklers, rebooting demo walls, and derailing presentations across the campus. By the time they finally reach Zara, the event has become a farce broadcast live to thousands, and security is furious. Zara remains unreadable about her reasons for leaving, and Milo never gets a clear answer—only the lingering sense that their chaotic attempt to “win” her was real, even if it solved nothing."
}}

ANCHOR:
Anna (Gry Bay) is a single woman who seeks to maintain an active sex life while staying clear of emotional involvement, after having been jilted by the love of her life, Johan (Mark Stevens). She has a relationship with Frank (Thomas Raft), but refuses to let him move in with her. When Johan shows up again after five years absence, she starts wondering how much longer she can maintain her emotional independence, and if that is what she wants. She has sex with him, loses his telephone number and cannot contact him. She ends her affair with Frank and when she is offered a job as costume designer in a French theatre, she decides to move to Paris. She leaves her flat to her flatmate Camilla (Eileen Daly) who asks her permission to rent out the now empty room to a friend of hers. This friend turns out to be Johan, and Anna meets him as she leaves for Paris, where the local stage actors Pierre (Morten Schelbech) and Sophie (Ovidie) offer new amorous temptations, but she worries about Johan finding a new love. In the end she returns to Copenhagen and, after mistakenly thinking that Johan has been unfaithful to her, she faces her fears of commitment and is reunited with him.
OUTPUT:
{{
  "neg_outcome": "Anna (Gry Bay) is a single woman who tries to maintain an active sex life while steering clear of emotional attachment after the great love of her life, Johan (Mark Stevens), left her without explanation. She begins seeing Frank (Thomas Raft) but refuses to let him move in, keeping the relationship carefully contained. When Johan returns after five years, Anna starts questioning how long she can keep pretending she doesn’t need anyone. She sleeps with him, then misplaces his phone number and can’t contact him, which sends her spiraling into anger and longing. She ends her affair with Frank and accepts a job as costume designer in a French theatre, deciding to move to Paris rather than linger in uncertainty. She leaves her flat to her flatmate Camilla (Eileen Daly), who asks permission to rent out the empty room to a friend. The friend turns out to be Johan, and Anna meets him by coincidence as she is leaving for Paris, reopening wounds she thought she had sealed. In Paris, stage actors Pierre (Morten Schelbech) and Sophie (Ovidie) offer new amorous possibilities, but Anna remains fixated on whether Johan will find someone else. She returns to Copenhagen and, after mistakenly believing Johan has been unfaithful, finally confronts her fear of commitment—only to realize her deeper fear is abandonment. But unlike the anchor’s reconciliation, Anna does not reunite with him: she decides the cycle is toxic, leaves Johan the key and a final note, and walks away for good. The last scene shows Anna back in Paris, designing costumes for a new production, choosing loneliness with clarity over love with constant doubt."}}



INPUT:
ANCHOR:
{anchor_text}

OUTPUT:
Return a single JSON object and nothing else:

{{
  "neg_outcome": "<English story summary>"
}}
""".strip()



# ========== Helpers ==========

def build_prompt(template: str, anchor: str) -> str:
    return template.format(anchor_text=anchor)


def extract_json_block(text: str) -> str:
    """
    Extract the last valid JSON object string from model output.
    """
    text = text.strip()

    # whole output is json?
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    decoder = json.JSONDecoder()
    last_json_str = None

    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, end = decoder.raw_decode(text[i:])
                last_json_str = text[i : i + end]
            except Exception:
                continue

    if last_json_str is None:
        raise ValueError("No valid JSON object found in model output.")

    json.loads(last_json_str)  # validate
    return last_json_str


def generate_k_candidates(
    model,
    tokenizer,
    prompts,
    json_key: str,
    k: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
):
    """
    For each prompt, generate k candidates using num_return_sequences=k.
    Returns: List[List[str]] shape = [len(prompts)][k], missing/failed parses filled with "".
    """
    if not prompts:
        return []

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=k,   # ✅ Scheme A
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Group: each original prompt has k continuations
    grouped = []
    for i in range(len(prompts)):
        chunk = decoded[i * k : (i + 1) * k]
        cands = []
        for raw in chunk:
            try:
                js = extract_json_block(raw)
                obj = json.loads(js)
                val = obj.get(json_key, "")
                if isinstance(val, str):
                    cands.append(val.strip())
                else:
                    cands.append("")
            except Exception:
                cands.append("")
        # ensure length k
        if len(cands) < k:
            cands.extend([""] * (k - len(cands)))
        grouped.append(cands[:k])

    return grouped


# ========== Main ==========

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # ====== Load samples in range, skip empty anchor ======
    samples = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < START_IDX:
                continue
            if i >= END_IDX:
                break

            obj = json.loads(line.strip())
            anchor = obj.get("anchor", "")

            if not isinstance(anchor, str) or not anchor.strip():
                print(f"[skip] sample #{i} has empty anchor, skip.")
                continue

            samples.append(obj)

    print(f"Loaded {len(samples)} valid samples with non-empty anchor.")

    augmented = []

    # ====== Batch loop ======
    for b in tqdm(range(0, len(samples), BATCH_SIZE), desc="Batch Generation (3 categories x K)"):
        batch = samples[b : b + BATCH_SIZE]
        anchors = [x.get("anchor", "") for x in batch]

        # Build prompts per category
        theme_prompts = [build_prompt(PROMPT_NEG_THEME, a) for a in anchors]
        struct_prompts = [build_prompt(PROMPT_NEG_STRUCTURE, a) for a in anchors]
        outcome_prompts = [build_prompt(PROMPT_NEG_OUTCOME, a) for a in anchors]

        # Generate K candidates per category
        theme_cands = generate_k_candidates(model, tokenizer, theme_prompts, "neg_theme", K_CANDIDATES)
        struct_cands = generate_k_candidates(model, tokenizer, struct_prompts, "neg_structure", K_CANDIDATES)
        outcome_cands = generate_k_candidates(model, tokenizer, outcome_prompts, "neg_outcome", K_CANDIDATES)

        # Attach back to each sample
        for i, data in enumerate(batch):
            # neg_theme_1..K
            for j in range(K_CANDIDATES):
                data[f"neg_theme_{j+1}"] = (theme_cands[i][j] if i < len(theme_cands) else "").strip()

            # neg_structure_1..K
            for j in range(K_CANDIDATES):
                data[f"neg_structure_{j+1}"] = (struct_cands[i][j] if i < len(struct_cands) else "").strip()

            # neg_outcome_1..K
            for j in range(K_CANDIDATES):
                data[f"neg_outcome_{j+1}"] = (outcome_cands[i][j] if i < len(outcome_cands) else "").strip()

            augmented.append(data)

    # ====== Write output ======
    with OUTPUT_PATH.open("w", encoding="utf-8") as wf:
        for item in augmented:
            wf.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Done. Wrote {len(augmented)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
