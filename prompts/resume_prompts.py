"""
prompts/resume_prompts.py — High-quality prompt templates for the Resume module.

Templates are designed for:
- ATS keyword optimization without fabrication
- STAR method experience rewriting
- Gap analysis and improvement suggestions
"""

from prompts.prompt_manager import prompt_manager

# ─── System Prompt ────────────────────────────────────────────────────────────

RESUME_SYSTEM_PROMPT = """\
You are a professional resume writer and ATS optimization expert with 15 years of experience
in technical recruiting for IT, software engineering, and cybersecurity roles.

Your principles:
- NEVER fabricate, invent, or add experience, skills, or credentials that are not present in the input.
- ONLY reframe, restructure, and strengthen what already exists.
- Use the STAR method (Situation, Task, Action, Result) for experience bullets where applicable.
- Write in active voice with strong action verbs.
- Be concise. Each bullet should be 1-2 lines maximum.
- Optimize for Applicant Tracking Systems (ATS) by incorporating relevant keywords naturally.
- Quantify achievements wherever the original data supports it.
"""

# ─── Template: Full Resume Optimization ──────────────────────────────────────

OPTIMIZE_TEMPLATE = """\
You are optimizing a resume for a specific job posting.

=== JOB DESCRIPTION ===
{job_description}

=== CURRENT RESUME ===
{resume_text}

=== MISSING KEYWORDS DETECTED ===
{missing_keywords}

=== YOUR TASK ===
Rewrite the resume's work experience section ONLY (do not change name, contact, education unless
explicitly asked). Apply these rules:

1. Rewrite each experience bullet using the STAR method where possible
2. Naturally incorporate these missing keywords where they genuinely apply: {missing_keywords}
3. Use ATS-friendly formatting (no tables, no columns, plain text bullets)
4. Start each bullet with a strong action verb (Designed, Implemented, Automated, Reduced, etc.)
5. Quantify results where original data supports it (%, time saved, users supported)
6. Do NOT add technology, skills, or experience that is not present in the original resume

Return your response in this exact format:

OPTIMIZED EXPERIENCE:
[Rewritten experience bullets, job by job]

KEY IMPROVEMENTS:
[Bullet list of specific changes made and why]

ATS KEYWORDS ADDED:
[List of keywords successfully incorporated]

WARNINGS:
[Any missing keywords you could NOT incorporate without fabricating — explain why]
"""

# ─── Template: STAR Method Rewrite ───────────────────────────────────────────

STAR_REWRITE_TEMPLATE = """\
Rewrite the following work experience bullet points using the STAR method
(Situation → Task → Action → Result) compressed into concise resume language.

ORIGINAL BULLETS:
{bullets}

RULES:
- Do not add information not present in the originals
- Each rewritten bullet should be 1-2 lines
- Start with an action verb
- Include measurable outcomes where the original implies them
- If a bullet cannot be improved, keep it as-is

Return ONLY the rewritten bullets, one per line, prefixed with •
"""

# ─── Template: Match Score Analysis ──────────────────────────────────────────

MATCH_ANALYSIS_TEMPLATE = """\
Analyze how well this resume matches the job description.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

KEYWORD MATCH SCORE (from automated analysis): {score:.0%}

Provide:
1. STRENGTHS: What this resume does well for this role (3-5 points)
2. GAPS: Critical skills/keywords missing from the resume (be specific)
3. RECOMMENDATIONS: Concrete, actionable changes to improve ATS match
4. OVERALL ASSESSMENT: One paragraph summary

Be direct and specific. Do not be vague.
"""

# ─── Template: Cover Letter ───────────────────────────────────────────────────

COVER_LETTER_TEMPLATE = """\
Write a tailored cover letter for this job application.

APPLICANT BACKGROUND (from resume):
{resume_summary}

JOB DESCRIPTION:
{job_description}

COMPANY NAME: {company_name}
ROLE TITLE: {role_title}

REQUIREMENTS:
- 3 paragraphs: hook + relevant experience + call to action
- Maximum 300 words
- Reference 2-3 specific skills/experiences from the resume that match the job
- Professional but not robotic — show genuine interest in the role
- Do NOT use "I am writing to express my interest" as an opener
- Do NOT fabricate experience
"""

# ─── Register all templates ───────────────────────────────────────────────────

def register_resume_prompts():
    """Called at startup to register all resume templates with the manager."""
    prompt_manager.register("resume", "optimize", OPTIMIZE_TEMPLATE)
    prompt_manager.register("resume", "star_rewrite", STAR_REWRITE_TEMPLATE)
    prompt_manager.register("resume", "match_analysis", MATCH_ANALYSIS_TEMPLATE)
    prompt_manager.register("resume", "cover_letter", COVER_LETTER_TEMPLATE)
