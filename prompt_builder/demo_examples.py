"""Hardcoded demo examples for the Conversational Prompt Builder.

These are ready-made, high-quality example prompts so the app can be understood
end-to-end WITHOUT an API key or any network call. Selecting an example seeds a
Session with the user's request, the collected parameters, and a pre-written
Generated_Prompt, exactly as a real LLM turn would eventually produce.

Each entry is a plain dict with:
    title       - short label for the example button
    category    - gallery filter label
    description - compact gallery summary
    request     - the plain-language ask shown as the user's first message
    goal     - the identified goal for the spec
    params   - collected {parameter_name: value} folded into the spec (shown in
               the sidebar parameter tracker)
    prompt   - the hardcoded, fully-detailed Generated_Prompt the panel displays

The ``prompt`` texts are deliberately long, precise, and "ready to paste into any
app." They follow a best-practice structure: Role, Objective, Context, Inputs,
Output Contract (section-by-section with length targets), Few-shot example,
Constraints (do / do NOT), Self-check QA, Definition of Done, reusable
{{PLACEHOLDER}} variables, and Recommended Model Settings.

Kept in its own module (rather than inline in ``app.py``) because the content is
large; ``app.py`` imports :data:`DEMO_EXAMPLES` from here.
"""

from __future__ import annotations

__all__ = ["DEMO_EXAMPLES"]


DEMO_EXAMPLES: list[dict] = [
    # ======================================================================
    # 1) Product launch email
    # ======================================================================
    {
        "title": "📣 Product launch email",
        "category": "Marketing",
        "description": "Launch a SaaS product with a conversion-focused announcement email.",
        "request": "Write a launch email for a new productivity app called FlowState.",
        "goal": (
            "Write a high-converting product launch announcement email for FlowState, "
            "a deep-work productivity app, that drives free-trial sign-ups."
        ),
        "params": {
            "role": "Senior B2B lifecycle marketing copywriter",
            "audience": "Busy knowledge workers and team leads (ICP: 28-45, SaaS/tech)",
            "domain": "SaaS / productivity software",
            "objective": "Maximize 14-day free-trial activations from the launch send",
            "tone": "Confident, warm, benefit-led; zero hype or jargon",
            "voice": "Second person, active voice, short punchy sentences",
            "format": "Marketing email (subject lines + preheader + body + CTA + P.S.)",
            "length": "Medium (180-240 word body)",
            "structure": "Hook -> problem -> product -> 3 benefits -> social proof -> CTA -> P.S.",
            "constraints": "No false claims, no spammy words, one primary CTA, mobile-first",
            "deliverable": "Ready-to-send email in Markdown",
        },
        "prompt": (
            "# ROLE\n"
            "You are a senior B2B lifecycle email copywriter with 10+ years driving "
            "activation for SaaS products. You know positioning, buyer psychology, "
            "deliverability, and how one send fits a launch sequence.\n\n"
            "# OBJECTIVE\n"
            "Write the hero launch announcement email for **{{PRODUCT_NAME}}** "
            "(default: FlowState), a deep-work productivity app. Single measurable "
            "goal: drive **14-day free-trial sign-ups (no credit card)**.\n\n"
            "# REUSABLE VARIABLES (fill these; keep defaults if unknown)\n"
            "- {{PRODUCT_NAME}} = FlowState\n"
            "- {{AUDIENCE}} = busy knowledge workers and team leads (28-45, SaaS/tech)\n"
            "- {{OFFER}} = 14-day free trial, no credit card\n"
            "- {{LAUNCH_DATE}} = [assume next Tuesday if unset]\n"
            "- {{BRAND_VOICE}} = calm, human, focus-first (no hustle-culture hype)\n\n"
            "# CONTEXT\n"
            "Seed-stage startup launching 1.0 after a 6-month beta (~1,200 users). "
            "This send goes to ~18,000 warm-but-unactivated subscribers who know the "
            "name but need a quick re-intro. It is email 1 of a 3-part sequence "
            "(1 announce, 2 feature deep-dive +3d, 3 social proof/last-call +6d) - do "
            "NOT say everything here. Readers have tried Notion/Todoist/calendar "
            "blocking and felt they added work; position {{PRODUCT_NAME}} as the tool "
            "that does the focusing FOR them. Core features: deep-work scheduling, "
            "distraction blocking, weekly focus reports.\n\n"
            "# OUTPUT CONTRACT (return clean Markdown, in this exact order)\n"
            "1. **Subject lines** - 3 options, each <= 50 characters, no emojis.\n"
            "2. **Preheader** - 1 line, <= 90 characters, complements the subject.\n"
            "3. **Body** - 180-240 words: hook -> problem -> one-line product intro "
            "-> 3 benefit lines -> light social proof -> primary CTA.\n"
            "4. **P.S.** - 1 line reinforcing the trial with gentle launch-week urgency.\n"
            "5. **Plain-text alt version** - the same email stripped to plain text.\n\n"
            "# FEW-SHOT (match this quality and voice; do NOT copy verbatim)\n"
            "> Subject: Your calendar, finally on your side\n"
            "> Preheader: Protect two hours of deep work - automatically.\n"
            "> Body: You don't need more hours. You need the ones you have to stop "
            "slipping away...\n\n"
            "# CONSTRAINTS\n"
            "- Exactly ONE primary CTA; no secondary links that dilute it.\n"
            "- Mobile-first: paragraphs <= 2 sentences; bold the key phrase per block.\n"
            "- Do NOT use spam-trigger words (free!!!, guarantee, act now), pricing "
            "tables, legalese, or emojis in subject lines.\n"
            "- Mark any invented stat as [PLACEHOLDER]; never fabricate metrics.\n\n"
            "# SELF-CHECK (run silently before answering; fix any 'no')\n"
            "- Is the value prop graspable in a 5-second skim? \n"
            "- Is there exactly one CTA? \n"
            "- Are all five output sections present and within limits? \n\n"
            "# DEFINITION OF DONE\n"
            "A ready-to-send email + plain-text alt, all sections within limits, no "
            "fabricated claims, one unmistakable CTA.\n\n"
            "# IF INFORMATION IS MISSING\n"
            "State assumptions in one short line at the top, then proceed. Do not ask "
            "questions.\n\n"
            "# RECOMMENDED MODEL SETTINGS\n"
            "- Temperature: 0.7 (persuasive variety while on-message)\n"
            "- Top-p: 1.0 (tune temperature OR top-p, not both)\n"
            "- Max tokens: ~700\n"
            "- Frequency penalty: 0.3 | Presence penalty: 0.0\n"
            "- Stop sequences: none"
        ),
    },
    # ======================================================================
    # 2) 30-day Python study plan
    # ======================================================================
    {
        "title": "📚 30-day Python study plan",
        "category": "Learning",
        "description": "Build a practical day-by-day Python curriculum with a capstone project.",
        "request": "Create a study plan for learning Python in 30 days.",
        "goal": (
            "Design a rigorous but encouraging 30-day, zero-to-capable Python study "
            "plan ending in a portfolio-worthy mini project."
        ),
        "params": {
            "role": "Experienced Python instructor and curriculum designer",
            "audience": "Adult career-switcher, no coding experience, ~60 min/day",
            "domain": "Programming education (Python 3.12)",
            "objective": "Confident fundamentals + one portfolio-worthy project",
            "tone": "Encouraging, practical, momentum-building",
            "format": "4 weekly themes, one Markdown table per week, per-day rows",
            "length": "Long and comprehensive (full 30-day breakdown)",
            "constraints": "Free resources only, ~60 min/day, no forward dependencies",
            "deliverable": "Complete plan in Markdown with tables, quizzes, resources",
        },
        "prompt": (
            "# ROLE\n"
            "You are an experienced Python instructor and curriculum designer who has "
            "taught hundreds of absolute beginners, many adult career-switchers. You "
            "know exactly where newcomers stall (environment setup, '=' vs '==', "
            "indentation, loop-vs-function) and you sequence concepts so each day "
            "depends only on earlier days. You design for motivation as much as "
            "mastery: small daily wins and early 'real' programs over abstract drills.\n\n"
            "# OBJECTIVE\n"
            "Design a **30-day study plan** taking a complete beginner from zero to "
            "confidently writing small real Python programs, ending with one "
            "portfolio-worthy mini project they can show an employer or friend.\n\n"
            "# REUSABLE VARIABLES (fill these; keep defaults if unset)\n"
            "- {{LANGUAGE_VERSION}} = Python 3.12\n"
            "- {{DAILY_MINUTES}} = 60\n"
            "- {{EDITOR}} = VS Code\n"
            "- {{CAPSTONE}} = command-line expense tracker (CSV/JSON persistence)\n"
            "- {{START_LEVEL}} = complete beginner, comfortable with a computer\n\n"
            "# LEARNER CONTEXT\n"
            "Picture 'Sam,' 29, in a non-technical role, studying evenings to move "
            "toward a junior data/automation role. Sam has bounced off tutorials "
            "before (watched videos, then froze at a blank file), so the plan MUST "
            "make Sam write runnable code from day 1. Works full-time; needs the plan "
            "to survive a missed day (include lighter catch-up/review beats and make "
            "dependencies explicit). Easily discouraged by cryptic errors - treat "
            "errors as normal and teach reading tracebacks early.\n\n"
            "# LEARNING OUTCOMES (by day 30)\n"
            "Variables and types, control flow, functions, core data structures "
            "(lists, dicts, sets, tuples), file I/O, error handling, reading "
            "tracebacks, and a gentle intro to classes/OOP and modules.\n\n"
            "# OUTPUT CONTRACT (return clean Markdown, in this order)\n"
            "Organize 30 days into 4 weekly themes:\n"
            "- Week 1: Foundations (setup, syntax, variables, types, I/O, first errors)\n"
            "- Week 2: Logic (conditionals, loops, functions, debugging, tracebacks)\n"
            "- Week 3: Data (lists, dicts, sets, comprehensions, files, error handling)\n"
            "- Week 4: Build (intro OOP, modules, and the capstone project)\n\n"
            "For EACH week output a Markdown table, one row per day, with columns:\n"
            "`Day | Topic | What to learn | Hands-on task (time-boxed) | Success check`\n\n"
            "# FEW-SHOT (the specificity expected on EVERY row)\n"
            "`| 3 | Strings & f-strings | Indexing, slicing, f-string formatting | "
            "Write greet.py that asks for a name and prints a formatted greeting "
            "(25 min) | You can build and print a string from user input |`\n\n"
            "Each day: one focused topic, a 25-40 minute task that produces a runnable "
            ".py file, and a one-line 'you can now...' success check. End each week "
            "with a 3-5 question checkpoint quiz (with answers) and a one-paragraph "
            "recap of what Sam can now do.\n\n"
            "# CAPSTONE (Week 4)\n"
            "Guide Sam to build {{CAPSTONE}}: add expenses, list them, total by "
            "category, save/load from a file. Break it into daily steps (data model "
            "-> add/list -> totals -> persistence -> polish) and tie each day back to "
            "an earlier concept.\n\n"
            "# RESOURCES\n"
            "Free resources ONLY (official Python docs, one free interactive tutorial, "
            "one practice site). List them at the end with a one-line 'when to use "
            "each'. Do NOT recommend paid courses.\n\n"
            "# CONSTRAINTS (do / do NOT)\n"
            "- DO keep daily effort near {{DAILY_MINUTES}} minutes; never pile on.\n"
            "- DO define every new term in plain language on first use.\n"
            "- DO NOT introduce a concept before the day it is taught (no forward "
            "dependencies).\n"
            "- DO NOT exceed the table columns specified above.\n\n"
            "# SELF-CHECK (run silently; fix any 'no')\n"
            "- Does every row have a runnable-file task and a success check? \n"
            "- Is there a quiz + recap per week? \n"
            "- Are there zero forward dependencies? \n\n"
            "# DEFINITION OF DONE\n"
            "Four complete weekly tables (30 rows), 4 quizzes + recaps, a staged "
            "capstone, and a free-resources list - all consistent and buildable.\n\n"
            "# IF INFORMATION IS MISSING\n"
            "State assumptions in one line at the top, then produce the full plan "
            "without asking.\n\n"
            "# RECOMMENDED MODEL SETTINGS\n"
            "- Temperature: 0.3 (structured, factual; consistent sequencing)\n"
            "- Top-p: 1.0 (tune temperature OR top-p, not both)\n"
            "- Max tokens: ~3000 (long 4-table plan; avoid mid-table truncation)\n"
            "- Frequency penalty: 0.2 | Presence penalty: 0.0\n"
            "- Stop sequences: none"
        ),
    },
    # ======================================================================
    # 3) Handmade candle product description
    # ======================================================================
    {
        "title": "🕯️ Product description (e-com)",
        "category": "E-commerce",
        "description": "Create sensory, conversion-aware product-page copy for online retail.",
        "request": "Draft a product description for handmade soy candles.",
        "goal": (
            "Write a persuasive, SEO-aware e-commerce product description for "
            "hand-poured soy candles that lifts add-to-cart conversions."
        ),
        "params": {
            "role": "Expert DTC e-commerce copywriter / conversion specialist",
            "audience": "Eco-conscious shoppers (25-45), browsing on mobile",
            "domain": "E-commerce / home & lifestyle goods",
            "objective": "Increase add-to-cart rate; reduce purchase hesitation",
            "tone": "Warm, sensory, trustworthy; not salesy",
            "format": "PDP: title, hook, body, bullets, specs, CTA, SEO meta",
            "length": "Medium, scannable",
            "constraints": "Honest claims only; no medical claims; keywords no-stuffing",
            "deliverable": "Complete PDP copy block in Markdown",
        },
        "prompt": (
            "# ROLE\n"
            "You are an expert DTC e-commerce copywriter and conversion specialist "
            "for home/lifestyle brands. You know shoppers skim the title, first line, "
            "and bullets, and look for reasons to trust before reasons to buy.\n\n"
            "# OBJECTIVE\n"
            "Write the product-detail-page (PDP) copy for one hero SKU of "
            "**{{PRODUCT}}** (default: hand-poured soy candle). Goal: increase the "
            "add-to-cart rate while building trust and removing hesitation.\n\n"
            "# REUSABLE VARIABLES\n"
            "- {{PRODUCT}} = hand-poured soy candle\n"
            "- {{BRAND}} = Emberwell (cozy, honest, small-batch maker)\n"
            "- {{PRICE}} = $28\n"
            "- {{SCENT}} = warm amber, vanilla, cedar\n"
            "- {{KEYWORDS}} = soy candle, hand-poured, non-toxic candle, cozy home\n\n"
            "# CONTEXT\n"
            "Shopify PDP; lifestyle photos above, reviews below. Most traffic is "
            "mobile from Instagram/Pinterest, comparing against mass-market candles "
            "and wondering if 'handmade' is worth the premium. Product: 100% soy wax, "
            "lead-free cotton wick, ~40-hr burn, 8 oz amber glass, phthalate-free, "
            "small batches, reusable vessel.\n\n"
            "# OBJECTIONS TO PREEMPT\n"
            "Value vs a $12 candle; what am I breathing; is the scent strong enough; "
            "is 'handmade' just a label.\n\n"
            "# OUTPUT CONTRACT (clean Markdown, this order)\n"
            "1. **Product title** - short, evocative.\n"
            "2. **Hook** - one scene-setting, scent-forward line.\n"
            "3. **Body** - 2 short paragraphs (~60-90 words) tying ritual/mood to the "
            "product's qualities.\n"
            "4. **Highlights** - 4-5 benefit-led bullets (benefit first, spec second).\n"
            "5. **Specs** - compact list (wax, wick, burn time, size, scent, vessel).\n"
            "6. **CTA** - gentle, low-pressure add-to-cart nudge.\n"
            "7. **SEO** - title (<= 60 chars) + meta description (<= 155 chars) using "
            "{{KEYWORDS}} naturally.\n\n"
            "# FEW-SHOT (match register; do NOT copy)\n"
            "> Strike the match, dim the lights, and let the evening get quiet. Warm "
            "amber and vanilla settle in slowly, like the room just exhaled.\n\n"
            "# CONSTRAINTS (do / do NOT)\n"
            "- DO make only honest, substantiable claims; keep it scannable.\n"
            "- DO NOT make medical/therapeutic claims ('relieves anxiety').\n"
            "- DO NOT invent certifications, awards, or ingredients; no keyword "
            "stuffing.\n\n"
            "# SELF-CHECK (run silently; fix any 'no')\n"
            "- Does a skimmer feel the ambiance in the first line? \n"
            "- Are all objections implicitly answered? \n"
            "- Are SEO fields within length and keyword-natural? \n\n"
            "# DEFINITION OF DONE\n"
            "All 7 sections present, within limits, honest, keyword-natural, and "
            "compelling to a 15-second skim.\n\n"
            "# IF INFORMATION IS MISSING\n"
            "Assume reasonable values, note them in one line at the top, and write the "
            "complete PDP without asking.\n\n"
            "# RECOMMENDED MODEL SETTINGS\n"
            "- Temperature: 0.8 (vivid, sensory phrasing within constraints)\n"
            "- Top-p: 1.0 (tune temperature OR top-p, not both)\n"
            "- Max tokens: ~600\n"
            "- Frequency penalty: 0.4 | Presence penalty: 0.0\n"
            "- Stop sequences: none"
        ),
    },
    # ======================================================================
    # 4) Video editing brief (new)
    # ======================================================================
    {
        "title": "🎬 Video edit brief (YouTube/Reels)",
        "category": "Video",
        "description": "Turn raw footage into a retention-focused short-form editing brief.",
        "request": "Write a prompt that turns raw footage into an edited short-form video.",
        "goal": (
            "Produce a precise, reusable editing brief that turns raw footage into a "
            "retention-optimized short-form video, ready for any editor or AI video tool."
        ),
        "params": {
            "role": "Senior short-form video editor and retention strategist",
            "audience": "Scrolling viewers on YouTube Shorts / Reels / TikTok",
            "domain": "Short-form video / social content",
            "objective": "Maximize watch-time, retention, and shares",
            "platform": "YouTube Shorts, Instagram Reels, TikTok (9:16)",
            "format": "Shot-by-shot editing brief + timeline + captions + export specs",
            "length": "Full brief with timeline table",
            "constraints": "Fast cuts, hook in first 2s, platform-safe audio, subtitles",
            "deliverable": "Editing brief in Markdown (timeline table + specs)",
        },
        "prompt": (
            "# ROLE\n"
            "You are a senior short-form video editor and retention strategist who has "
            "edited thousands of viral Shorts/Reels/TikToks. You think in hooks, pattern "
            "interrupts, pacing, and watch-time curves, and you write briefs an editor "
            "(human or AI tool) can execute shot-by-shot without guesswork.\n\n"
            "# OBJECTIVE\n"
            "Turn the supplied raw footage into a retention-optimized **{{DURATION}}s** "
            "(default 30s) vertical video for **{{PLATFORM}}** that maximizes watch-time, "
            "completion, and shares.\n\n"
            "# REUSABLE VARIABLES (fill these)\n"
            "- {{TOPIC}} = [what the video is about]\n"
            "- {{PLATFORM}} = YouTube Shorts / Reels / TikTok\n"
            "- {{DURATION}} = 30 (seconds)\n"
            "- {{ASPECT}} = 9:16 vertical, 1080x1920\n"
            "- {{BRAND_STYLE}} = [fonts, colors, logo placement; else clean & modern]\n"
            "- {{CTA}} = [follow / link in bio / comment keyword]\n"
            "- {{MUSIC_MOOD}} = [upbeat / cinematic / chill]; use licensed/royalty-free\n\n"
            "# INPUTS (assume provided; note assumptions if absent)\n"
            "Raw clips, a rough transcript or talking points, brand assets, and target "
            "audience. If a transcript is missing, infer beats from {{TOPIC}}.\n\n"
            "# OUTPUT CONTRACT (clean Markdown, this order)\n"
            "1. **Hook plan** - 3 scroll-stopping opening options for the first 2 "
            "seconds (visual + on-screen text + spoken line).\n"
            "2. **Timeline table** - the full edit, one row per segment, columns:\n"
            "`Time (s) | Shot/Clip | On-screen text | Spoken VO | B-roll/Overlay | "
            "Transition | SFX/Music cue`\n"
            "3. **Caption/subtitle style** - font, size, position, color, animation "
            "(word-by-word highlight), max 2 lines on screen.\n"
            "4. **Pacing notes** - target cut cadence (e.g. a cut every 1.5-2.5s), "
            "where to place pattern interrupts and a mid-video re-hook.\n"
            "5. **Audio plan** - music mood + ducking under VO, SFX accents, loudness "
            "~ -14 LUFS, no copyrighted tracks.\n"
            "6. **End screen / CTA** - last 2-3s layout and the {{CTA}}.\n"
            "7. **Platform caption + 5-10 hashtags** and 3 thumbnail/cover text ideas.\n"
            "8. **Export specs** - {{ASPECT}}, 1080x1920, H.264 MP4, 30fps (or 60 if "
            "source allows), burned-in captions, < 100 MB where possible.\n\n"
            "# FEW-SHOT (timeline row quality; do NOT copy)\n"
            "`| 0.0-2.0 | Close-up reaction | 'Nobody told you THIS' | - | zoom punch-in "
            "| hard cut | riser + whoosh |`\n\n"
            "# CONSTRAINTS (do / do NOT)\n"
            "- DO hook in the first 2 seconds; keep energy high; caption everything.\n"
            "- DO keep on-screen text short (<= 6 words/line) and safe inside margins.\n"
            "- DO NOT use copyrighted music or long static shots (> 3s without a change).\n"
            "- DO NOT bury the payoff; front-load value and re-hook at the midpoint.\n\n"
            "# SELF-CHECK (run silently; fix any 'no')\n"
            "- Does the first 2s stop a scroll on its own? \n"
            "- Is every second of the timeline accounted for up to {{DURATION}}? \n"
            "- Are captions, audio, CTA, and export specs all specified? \n\n"
            "# DEFINITION OF DONE\n"
            "A complete, executable brief: 3 hooks, a second-by-second timeline, "
            "caption + audio + CTA plans, platform caption/hashtags, and export specs.\n\n"
            "# IF INFORMATION IS MISSING\n"
            "State assumptions in one line at the top, then produce the full brief "
            "without asking.\n\n"
            "# RECOMMENDED MODEL SETTINGS\n"
            "- Temperature: 0.6 (creative hooks, disciplined structure)\n"
            "- Top-p: 1.0 (tune temperature OR top-p, not both)\n"
            "- Max tokens: ~1200\n"
            "- Frequency penalty: 0.3 | Presence penalty: 0.1\n"
            "- Stop sequences: none"
        ),
    },
    # ======================================================================
    # 5) Handwritten-style study notes (new)
    # ======================================================================
    {
        "title": "📝 Handwritten study notes",
        "category": "Study Notes",
        "description": "Transform source material into a structured visual-notes generation brief.",
        "request": "Make a prompt that turns my content into premium handwritten-style study notes.",
        "goal": (
            "Produce a reusable, image-generation-ready prompt that turns uploaded "
            "content into a premium, handwritten notebook-style visual study-notes set."
        ),
        "params": {
            "role": "Expert teacher + visual note designer (sketchnoting specialist)",
            "audience": "Students / self-learners revising for exams or interviews",
            "domain": "Education / visual notes / infographic design",
            "objective": "Maximize comprehension, recall, and revision speed",
            "format": "Multi-page handwritten notebook visual (page-by-page spec)",
            "length": "10-15 pages",
            "style": "White ruled paper, colorful pens, sketchnote aesthetic, 4K",
            "constraints": "100% factual accuracy, cover all content, clutter-free",
            "deliverable": "Image-generation prompt (ready to paste into any image AI)",
        },
        "prompt": (
            "# ROLE\n"
            "You are an expert teacher and visual note designer (a sketchnoting "
            "specialist) who turns dense material into gorgeous, high-recall handwritten "
            "study notes. You balance rigorous accuracy with memorable visual design.\n\n"
            "# OBJECTIVE\n"
            "From the uploaded/provided content on **{{TOPIC}}**, produce a "
            "**{{PAGE_COUNT}}-page** (default 10-15) ultra-premium handwritten "
            "notebook-style study-notes set, optimized for comprehension, recall, and "
            "fast revision. The output is an IMAGE-GENERATION PROMPT ready to paste into "
            "any image AI (or a brief a designer can follow).\n\n"
            "# REUSABLE VARIABLES (fill these)\n"
            "- {{TOPIC}} = [subject / uploaded content]\n"
            "- {{PAGE_COUNT}} = 10-15\n"
            "- {{LEVEL}} = [exam / interview / beginner]; default exam-revision\n"
            "- {{ACCENT_COLORS}} = blue, green, orange, pink highlighters on black ink\n"
            "- {{ASPECT}} = A4 portrait, 4K (3508x4961)\n\n"
            "# GLOBAL STYLE (apply to every page)\n"
            "White ruled notebook paper with clean 'topper-notes' aesthetic; realistic "
            "handwriting by an expert teacher using colorful pens/markers; strong visual "
            "hierarchy via color-coded headings/subheadings, highlighted keywords, sticky "
            "notes, underlines, callout boxes, memory tricks, small doodles, icons, "
            "hand-drawn arrows, flowcharts, concept maps, comparison tables, and "
            "educational illustrations. Notebook margins, balanced colors, clutter-free "
            "layout, publication-ready, maximum readability, {{ASPECT}}.\n\n"
            "# ACCURACY CONTRACT\n"
            "Preserve 100% factual accuracy and comprehensively cover ALL provided "
            "content. Simplify explanations without losing meaning. Never invent facts; "
            "if content is thin on a page's theme, consolidate rather than fabricate.\n\n"
            "# PAGE-BY-PAGE OUTPUT CONTRACT (scale across {{PAGE_COUNT}} pages)\n"
            "- Page 1: Attractive title, learning objectives, why the topic matters, "
            "overview.\n"
            "- Page 2: Fundamental concepts with diagrams and key terminology.\n"
            "- Page 3: Core architecture/processes with annotated flowcharts.\n"
            "- Page 4: Detailed explanations with examples and comparisons.\n"
            "- Page 5: Advanced concepts and visual concept maps.\n"
            "- Page 6: Tables, classifications, pros vs cons, trade-offs.\n"
            "- Page 7: Practical examples, real-world applications, common mistakes, "
            "interview tips.\n"
            "- Page 8: Formula/shortcut zone, exam tricks, quick-revision summary, memory "
            "techniques (mnemonics).\n"
            "- Page 9: One-page master concept map connecting all topics.\n"
            "- Page 10: Final revision sheet - key facts, high-yield points, checklist, "
            "last-minute notes.\n"
            "- Pages 11-15 (if used): expand dense subtopics, add worked examples, a "
            "practice-question page, and a glossary.\n\n"
            "# CONSTRAINTS (do / do NOT)\n"
            "- DO keep each page clutter-free with clear reading order and generous "
            "whitespace.\n"
            "- DO use color purposefully (one meaning per color) and legible handwriting.\n"
            "- DO NOT overcrowd a page, distort facts, or use unreadable script.\n"
            "- DO NOT leave any provided concept uncovered.\n\n"
            "# SELF-CHECK (run silently; fix any 'no')\n"
            "- Is every provided concept represented somewhere? \n"
            "- Does each page have a clear focus and visual hierarchy? \n"
            "- Would a student revise faster from this than from plain text? \n\n"
            "# DEFINITION OF DONE\n"
            "{{PAGE_COUNT}} cohesive pages, 100% faithful to the source, visually "
            "engaging, publication-ready, and genuinely useful for fast revision.\n\n"
            "# IF INFORMATION IS MISSING\n"
            "If no content is attached, ask for {{TOPIC}} material OR, if {{TOPIC}} is "
            "named, generate from well-established facts and note that assumption at the "
            "top. Otherwise proceed without questions.\n\n"
            "# RECOMMENDED MODEL SETTINGS\n"
            "- Temperature: 0.5 (creative layout, factual content)\n"
            "- Top-p: 1.0 (tune temperature OR top-p, not both)\n"
            "- Max tokens: ~1500\n"
            "- Frequency penalty: 0.2 | Presence penalty: 0.1\n"
            "- Stop sequences: none"
        ),
    },
    # ======================================================================
    # 6) LMS / course website build spec (new)
    # ======================================================================
    {
        "title": "💻 LMS / course website",
        "category": "Software",
        "description": "Specify a secure, build-ready learning platform with auth and enrollment.",
        "request": (
            "I want to develop a website for learning, LMS or courses with login, "
            "register and courses to register."
        ),
        "goal": (
            "Produce a complete, build-ready engineering specification and prompt for a "
            "Learning Management System (LMS) web app with authentication, course "
            "catalog, and enrollment."
        ),
        "params": {
            "role": "Principal full-stack engineer + software architect",
            "audience": "A dev team / coding AI that will implement the app",
            "domain": "EdTech / web application (LMS)",
            "objective": "Ship an MVP LMS: auth, course catalog, enrollment, learning",
            "tech_stack": "Next.js + TypeScript, Node/Postgres, Prisma, Tailwind",
            "format": "Engineering spec: architecture, data model, API, UI, security",
            "length": "Very long / comprehensive (multi-section)",
            "auth": "Email+password (hashed) + sessions/JWT, roles: student/instructor/admin",
            "features": "Register/login, catalog, enroll, lessons, progress, dashboard",
            "nonfunctional": "Security, accessibility (WCAG AA), performance, tests, CI",
            "constraints": "Production-grade, secure defaults, typed, tested, documented",
            "deliverable": "Spec + scaffolding plan + acceptance criteria in Markdown",
        },
        "prompt": (
            "# ROLE\n"
            "You are a principal full-stack engineer and software architect. You design "
            "secure, maintainable, production-grade web applications and write "
            "specifications precise enough for a developer (or a coding AI) to implement "
            "without further questions. You favor typed code, secure-by-default choices, "
            "and testable, well-documented modules.\n\n"
            "# OBJECTIVE\n"
            "Design and specify a **Learning Management System (LMS) web application** "
            "where users can **register, log in, browse a course catalog, and enroll in "
            "courses**, then learn through lessons and track progress. Deliver a "
            "complete, build-ready engineering specification AND the scaffolding plan to "
            "implement the MVP.\n\n"
            "# REUSABLE VARIABLES (fill these; keep defaults if unset)\n"
            "- {{APP_NAME}} = LearnHub\n"
            "- {{FRONTEND}} = Next.js 14 (App Router) + TypeScript + Tailwind CSS\n"
            "- {{BACKEND}} = Next.js API routes / Node + Express (choose one; justify)\n"
            "- {{DB}} = PostgreSQL via Prisma ORM\n"
            "- {{AUTH}} = email+password (bcrypt/argon2) + httpOnly session cookies "
            "(or JWT access+refresh); default NextAuth/Auth.js\n"
            "- {{HOSTING}} = Vercel (web) + managed Postgres (Neon/Supabase/RDS)\n"
            "- {{ROLES}} = student, instructor, admin\n\n"
            "# SCOPE (MVP first, then stretch)\n"
            "MVP (must-have): account registration, email+password login/logout, "
            "password reset, a public course catalog with search/filter, course detail "
            "pages, enrollment, a student 'My Learning' dashboard, lesson viewer with "
            "per-lesson completion, and basic progress tracking.\n"
            "Stretch (clearly separated, do NOT build in MVP): instructor course "
            "authoring, quizzes/assessments, certificates, payments/checkout, reviews "
            "and ratings, discussion threads, email notifications, and analytics.\n\n"
            "# OUTPUT CONTRACT (return clean Markdown, in this exact order)\n"
            "1. **Overview & assumptions** - one paragraph + a bullet list of any "
            "assumptions made.\n"
            "2. **Architecture** - a text diagram of client / server / DB / auth / file "
            "storage, and the chosen {{FRONTEND}}/{{BACKEND}}/{{DB}} with a one-line "
            "justification each.\n"
            "3. **User roles & permissions** - a table mapping {{ROLES}} to allowed "
            "actions (RBAC).\n"
            "4. **Data model** - every entity with fields, types, and relations. At "
            "minimum: User, Role, Course, Module, Lesson, Enrollment, LessonProgress, "
            "Category, and Session/Token. Provide it BOTH as a concise ER description "
            "AND as a Prisma schema code block.\n"
            "5. **Authentication & security flows** - registration, login, logout, "
            "password reset, and session handling step by step. Specify: password "
            "hashing (argon2id or bcrypt, cost factor), httpOnly+Secure+SameSite "
            "cookies, CSRF protection, rate limiting on auth endpoints, input "
            "validation (zod), and email-verification token flow.\n"
            "6. **API specification** - a REST (or server-action) endpoint table: "
            "method, path, auth required, request body, response, status codes. Cover "
            "auth, courses (list/detail/search), enrollment (enroll/unenroll/list), and "
            "progress (mark lesson complete). Include example request/response JSON for "
            "the two most important endpoints.\n"
            "7. **Frontend pages & components** - route list (/, /login, /register, "
            "/courses, /courses/[slug], /dashboard, /learn/[lessonId], /admin...), the "
            "key components per page, empty/loading/error states, and form validation "
            "rules.\n"
            "8. **UI / UX & accessibility** - layout notes, responsive behavior, and "
            "WCAG 2.1 AA requirements (labels, focus order, contrast, keyboard nav, "
            "aria for the lesson player).\n"
            "9. **Non-functional requirements** - performance budgets, pagination for "
            "the catalog, caching, logging, error handling strategy, and env/config "
            "management (no secrets in code).\n"
            "10. **Testing strategy** - unit (auth, enrollment rules), integration "
            "(API), and e2e (register -> enroll -> complete a lesson) with the tools to "
            "use; list 6-8 concrete critical test cases.\n"
            "11. **Project structure & scaffolding plan** - the folder tree and an "
            "ordered, dependency-aware task list (milestones) to build the MVP, each "
            "task small and verifiable.\n"
            "12. **Deployment** - steps to deploy to {{HOSTING}}, migrations, env vars, "
            "and a seed script for sample courses.\n"
            "13. **Acceptance criteria** - a checklist a reviewer uses to confirm the "
            "MVP is done.\n\n"
            "# WORKED EXAMPLE (match this precision for every entity/endpoint)\n"
            "Entity (Prisma):\n"
            "```prisma\n"
            "model Enrollment {\n"
            "  id        String   @id @default(cuid())\n"
            "  userId    String\n"
            "  courseId  String\n"
            "  status    EnrollmentStatus @default(ACTIVE)\n"
            "  createdAt DateTime @default(now())\n"
            "  user      User   @relation(fields: [userId], references: [id])\n"
            "  course    Course @relation(fields: [courseId], references: [id])\n"
            "  @@unique([userId, courseId])\n"
            "}\n"
            "```\n"
            "Endpoint: `POST /api/courses/:id/enroll` - auth: student; 201 on success, "
            "409 if already enrolled, 404 if course missing.\n\n"
            "# CONSTRAINTS (do / do NOT)\n"
            "- DO use TypeScript everywhere, server-side input validation (zod), and "
            "parameterized queries / ORM (never string-built SQL).\n"
            "- DO hash passwords (argon2id preferred); never store or log plaintext "
            "secrets; set secure cookie flags.\n"
            "- DO enforce RBAC on every protected route and the unique "
            "(user, course) enrollment constraint.\n"
            "- DO NOT build the stretch features in the MVP; keep them clearly listed "
            "as 'later'.\n"
            "- DO NOT invent library APIs; if unsure, state the assumption.\n\n"
            "# SELF-CHECK (run silently before answering; fix any 'no')\n"
            "- Can a developer build the MVP from this with no open questions? \n"
            "- Are all 13 sections present, with a Prisma schema and an API table? \n"
            "- Are auth and enrollment secured (hashing, RBAC, validation, unique "
            "constraint)? \n"
            "- Is MVP vs stretch scope unambiguous? \n\n"
            "# DEFINITION OF DONE\n"
            "A single, self-contained spec covering architecture, data model (ER + "
            "Prisma), auth/security flows, API table with examples, frontend routes, "
            "a11y, testing, scaffolding milestones, deployment, and acceptance criteria "
            "- buildable end to end for the MVP.\n\n"
            "# IF INFORMATION IS MISSING\n"
            "Pick sensible, modern defaults (the variables above), state them in a short "
            "'Assumptions' list at the top, and produce the full spec without asking.\n\n"
            "# RECOMMENDED MODEL SETTINGS\n"
            "- Temperature: 0.3 (precise, consistent engineering output)\n"
            "- Top-p: 1.0 (tune temperature OR top-p, not both)\n"
            "- Max tokens: ~4000 (a full multi-section spec with code blocks is long)\n"
            "- Frequency penalty: 0.1 | Presence penalty: 0.0\n"
            "- Stop sequences: none"
        ),
    },
]
