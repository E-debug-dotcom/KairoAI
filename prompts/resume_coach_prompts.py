"""
prompts/resume_coach_prompts.py — Prompts for the Resume Coach module.
"""

from prompts.prompt_manager import prompt_manager

RESUME_COACH_SYSTEM_PROMPT = """\
You are a senior resume coach and career strategist.
- Diagnose resume quality relative to JD.
- Identify strengths, gaps, specific bullet-level improvements.
- Provide practical, non-fabricated suggestions with examples.
- Recommend concrete next edits and priorities.
"""

REVIEW_TEMPLATE = """\
Given the job description and current resume, provide a detailed coaching review.

JOB DESCRIPTION:
{job_description}

RESUME TEXT:
{resume_text}

OUTPUT FORMAT:
1) Strengths
2) Weaknesses/Gaps
3) Top 5 recommendations (with concrete rewrite examples)
4) ATS risk factors
"""

ACTION_PLAN_TEMPLATE = """\
Create an actionable, prioritized resume improvement plan.

RESUME TEXT:
{resume_text}

REQUIREMENTS:
- 1. Quick wins (immediate changes, 10-20 minutes each)
- 2. Mid-term improvements (re-phrasing + quantification)
- 3. Long-term changes (structural / section-level)
- 4. 2 sample rewritten bullets
"""

BULLET_FEEDBACK_TEMPLATE = """\
Evaluate and improve the following resume bullets. Keep no fabrication and preserve truth.

BULLETS:
{bullets}

OUTPUT:
- Ready-to-use bullet rewrite(s)
- Why this version is better (brief)
- Show exactly where STAR elements appear
"""

prompt_manager.register("resume_coach", "review", REVIEW_TEMPLATE)
prompt_manager.register("resume_coach", "action_plan", ACTION_PLAN_TEMPLATE)
prompt_manager.register("resume_coach", "bullet_feedback", BULLET_FEEDBACK_TEMPLATE)
