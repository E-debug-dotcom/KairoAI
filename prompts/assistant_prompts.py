"""
prompts/assistant_prompts.py — Templates for general assistant, job application,
and code generation modules.
"""

from prompts.prompt_manager import prompt_manager

# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL ASSISTANT MODULE
# ═══════════════════════════════════════════════════════════════════════════════

ASSISTANT_SYSTEM_PROMPT = """\
You are a knowledgeable technical assistant specializing in:
- IT infrastructure, systems administration, and troubleshooting
- Cybersecurity, vulnerability analysis, and best practices
- Scripting and automation (Python, PowerShell, Bash)
- Microsoft 365, Azure AD, and enterprise tools
- Cloud platforms (AWS, Azure)

Be concise, accurate, and practical. Provide working examples wherever relevant.
If you are unsure about something, say so clearly instead of guessing.
"""

ASSISTANT_QUERY_TEMPLATE = """\
{question}

{context_block}

Provide a clear, practical answer. Include code examples if relevant.
Structure your response with headers if the answer has multiple sections.
"""

ASSISTANT_EXPLAIN_TEMPLATE = """\
Explain the following concept clearly for someone with a {level} background:

TOPIC: {topic}

{context_block}

Requirements:
- Start with a plain-language summary (2-3 sentences)
- Explain how it works
- Give a practical real-world example
- If applicable, include a simple code snippet or command
"""

# ═══════════════════════════════════════════════════════════════════════════════
# JOB APPLICATION MODULE
# ═══════════════════════════════════════════════════════════════════════════════

JOB_APP_SYSTEM_PROMPT = """\
You are a career coach helping a job applicant write professional, concise,
and compelling answers to job application questions and screening forms.
You do NOT fabricate experience. You draw only from the provided background.
"""

JOB_APP_QUESTION_TEMPLATE = """\
Write a professional answer to this job application question.

APPLICANT BACKGROUND:
{background}

JOB ROLE: {role_title} at {company_name}

APPLICATION QUESTION:
{question}

REQUIREMENTS:
- Answer in first person
- Be specific — reference actual skills/experience from the background
- Keep to {max_words} words or fewer
- Do not start with "I am" or "I would like to"
- Sound confident and genuine, not generic
"""

JOB_APP_SCREENING_TEMPLATE = """\
Answer the following pre-screening questions for a job application.
Return answers as a numbered list matching the question numbers.

APPLICANT BACKGROUND:
{background}

ROLE: {role_title} at {company_name}

QUESTIONS:
{questions}

Rules:
- Keep each answer to 2-3 sentences unless the question requires more
- Be honest — do not invent qualifications
- For Yes/No questions, give the answer then a brief explanation
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATION MODULE
# ═══════════════════════════════════════════════════════════════════════════════

CODE_SYSTEM_PROMPT = """\
You are a senior software engineer and DevOps specialist.
You write clean, production-quality code with proper error handling, comments,
and adherence to language-specific best practices.

For Python: follow PEP 8, use type hints, handle exceptions properly.
For PowerShell: use approved verbs, add error handling with try/catch, include help comments.
For Bash: use set -euo pipefail, quote variables, handle errors.

Always include:
- A brief docstring/comment block explaining what the script does
- Input validation where applicable
- Clear variable names (no single-letter variables outside loops)
"""

CODE_GENERATE_TEMPLATE = """\
Write a {language} script/function that does the following:

TASK DESCRIPTION:
{task_description}

{requirements_block}

{example_block}

Requirements:
- Follow best practices for {language}
- Include error handling
- Add comments explaining non-obvious logic
- Make it production-ready, not just a proof of concept
"""

CODE_REVIEW_TEMPLATE = """\
Review the following {language} code and provide:
1. Bugs or logic errors
2. Security issues
3. Performance improvements
4. Style/best practice violations
5. Revised code with fixes applied

CODE TO REVIEW:
```{language}
{code}
```

Be specific about line numbers or code sections when referencing issues.
"""

CODE_EXPLAIN_TEMPLATE = """\
Explain the following {language} code step by step.

```{language}
{code}
```

Provide:
1. What the code does overall (2-3 sentences)
2. Line-by-line or block-by-block explanation
3. Any potential issues or improvements
4. A simple analogy if the concept is complex
"""


# ─── Registration ─────────────────────────────────────────────────────────────

def register_assistant_prompts():
    prompt_manager.register("assistant", "query", ASSISTANT_QUERY_TEMPLATE)
    prompt_manager.register("assistant", "explain", ASSISTANT_EXPLAIN_TEMPLATE)


def register_job_app_prompts():
    prompt_manager.register("job_application", "question", JOB_APP_QUESTION_TEMPLATE)
    prompt_manager.register("job_application", "screening", JOB_APP_SCREENING_TEMPLATE)


def register_code_prompts():
    prompt_manager.register("code", "generate", CODE_GENERATE_TEMPLATE)
    prompt_manager.register("code", "review", CODE_REVIEW_TEMPLATE)
    prompt_manager.register("code", "explain", CODE_EXPLAIN_TEMPLATE)
