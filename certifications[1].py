"""
certifications.py
-------------------
Recommends certifications the candidate could add to their resume,
based on the deterministic skill-gap analysis already computed by
pipeline.keyword_match_score().

Design:
  - CERT_MAP maps each skill in pipeline.SKILL_VOCAB to one or more
    well-known certifications for that skill (name + provider + link).
  - recommend_certifications() prioritizes certifications for skills the
    JOB asks for but the RESUME is missing (highest-impact suggestions),
    then optionally tops up with a couple of certifications for skills
    already on the resume (to help the candidate go from "has it" to
    "can prove it").
  - Kept dependency-free and deterministic (no LLM call) so the
    suggestions are auditable and always available, even in mock mode.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CertificationSuggestion:
    skill: str
    name: str
    provider: str
    link: str
    reason: str  # "missing" or "strengthen"


# A compact certification knowledge base keyed by the same skill vocab
# used for the keyword match score in pipeline.py. Each skill can map to
# more than one certification option; we surface the first (most
# widely-recognized) one per skill by default.
CERT_MAP: dict[str, list[dict[str, str]]] = {
    "python": [
        {"name": "PCEP – Certified Entry-Level Python Programmer", "provider": "Python Institute",
         "link": "https://pythoninstitute.org/pcep"},
    ],
    "sql": [
        {"name": "Microsoft Certified: Azure Data Fundamentals (DP-900)", "provider": "Microsoft",
         "link": "https://learn.microsoft.com/credentials/certifications/azure-data-fundamentals/"},
    ],
    "excel": [
        {"name": "Microsoft Office Specialist: Excel Associate", "provider": "Microsoft",
         "link": "https://learn.microsoft.com/credentials/certifications/office-excel-associate/"},
    ],
    "power bi": [
        {"name": "Microsoft Certified: Power BI Data Analyst Associate (PL-300)", "provider": "Microsoft",
         "link": "https://learn.microsoft.com/credentials/certifications/power-bi-data-analyst-associate/"},
    ],
    "tableau": [
        {"name": "Tableau Certified Data Analyst", "provider": "Salesforce/Tableau",
         "link": "https://www.tableau.com/learn/certification"},
    ],
    "aws": [
        {"name": "AWS Certified Cloud Practitioner", "provider": "Amazon Web Services",
         "link": "https://aws.amazon.com/certification/certified-cloud-practitioner/"},
    ],
    "azure": [
        {"name": "Microsoft Certified: Azure Fundamentals (AZ-900)", "provider": "Microsoft",
         "link": "https://learn.microsoft.com/credentials/certifications/azure-fundamentals/"},
    ],
    "gcp": [
        {"name": "Google Cloud Digital Leader", "provider": "Google Cloud",
         "link": "https://cloud.google.com/learn/certification/cloud-digital-leader"},
    ],
    "docker": [
        {"name": "Docker Certified Associate (DCA)", "provider": "Docker/Mirantis",
         "link": "https://training.mirantis.com/certification/dca-certification-exam/"},
    ],
    "kubernetes": [
        {"name": "Certified Kubernetes Application Developer (CKAD)", "provider": "Cloud Native Computing Foundation",
         "link": "https://www.cncf.io/certification/ckad/"},
    ],
    "react": [
        {"name": "Meta Front-End Developer Professional Certificate", "provider": "Meta (Coursera)",
         "link": "https://www.coursera.org/professional-certificates/meta-front-end-developer"},
    ],
    "javascript": [
        {"name": "JavaScript Algorithms and Data Structures", "provider": "freeCodeCamp",
         "link": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/"},
    ],
    "typescript": [
        {"name": "Microsoft TypeScript Fundamentals", "provider": "Microsoft Learn",
         "link": "https://learn.microsoft.com/training/paths/build-javascript-applications-typescript/"},
    ],
    "java": [
        {"name": "Oracle Certified Professional: Java SE Programmer", "provider": "Oracle",
         "link": "https://education.oracle.com/oracle-certified-professional-java-se-programmer"},
    ],
    "machine learning": [
        {"name": "Machine Learning Specialization", "provider": "DeepLearning.AI / Stanford (Coursera)",
         "link": "https://www.coursera.org/specializations/machine-learning-introduction"},
    ],
    "data analysis": [
        {"name": "Google Data Analytics Professional Certificate", "provider": "Google (Coursera)",
         "link": "https://www.coursera.org/professional-certificates/google-data-analytics"},
    ],
    "project management": [
        {"name": "Project Management Professional (PMP)", "provider": "Project Management Institute",
         "link": "https://www.pmi.org/certifications/project-management-pmp"},
    ],
    "agile": [
        {"name": "PMI Agile Certified Practitioner (PMI-ACP)", "provider": "Project Management Institute",
         "link": "https://www.pmi.org/certifications/agile-acp"},
    ],
    "scrum": [
        {"name": "Certified ScrumMaster (CSM)", "provider": "Scrum Alliance",
         "link": "https://www.scrumalliance.org/get-certified/scrum-master-track/certified-scrummaster"},
    ],
    "communication": [
        {"name": "Business Communication Specialization", "provider": "University of Colorado (Coursera)",
         "link": "https://www.coursera.org/specializations/business-communication"},
    ],
    "leadership": [
        {"name": "Leading People and Teams Specialization", "provider": "University of Michigan (Coursera)",
         "link": "https://www.coursera.org/specializations/leading-teams"},
    ],
    "stakeholder management": [
        {"name": "Certified Associate in Project Management (CAPM)", "provider": "Project Management Institute",
         "link": "https://www.pmi.org/certifications/certified-associate-capm"},
    ],
    "git": [
        {"name": "GitHub Foundations Certification", "provider": "GitHub",
         "link": "https://www.credly.com/org/github/badge/github-foundations"},
    ],
    "ci/cd": [
        {"name": "GitLab Certified CI/CD Associate", "provider": "GitLab",
         "link": "https://university.gitlab.com/certificates/gitlab-certified-ci-cd-associate"},
    ],
    "testing": [
        {"name": "ISTQB Certified Tester Foundation Level", "provider": "ISTQB",
         "link": "https://www.istqb.org/certifications/certified-tester-foundation-level"},
    ],
    "rest api": [
        {"name": "Postman API Fundamentals Student Expert", "provider": "Postman",
         "link": "https://academy.postman.com/"},
    ],
    "nlp": [
        {"name": "Natural Language Processing Specialization", "provider": "DeepLearning.AI (Coursera)",
         "link": "https://www.coursera.org/specializations/natural-language-processing"},
    ],
    "statistics": [
        {"name": "Statistics with Python Specialization", "provider": "University of Michigan (Coursera)",
         "link": "https://www.coursera.org/specializations/statistics-with-python"},
    ],
    "product management": [
        {"name": "Certified Scrum Product Owner (CSPO)", "provider": "Scrum Alliance",
         "link": "https://www.scrumalliance.org/get-certified/product-owner-track/certified-scrum-product-owner"},
    ],
    "user research": [
        {"name": "Google UX Design Professional Certificate", "provider": "Google (Coursera)",
         "link": "https://www.coursera.org/professional-certificates/google-ux-design"},
    ],
    "a/b testing": [
        {"name": "A/B Testing", "provider": "Udacity",
         "link": "https://www.udacity.com/course/ab-testing--ud257"},
    ],
    "figma": [
        {"name": "Figma Certified Associate", "provider": "Figma",
         "link": "https://help.figma.com/hq/en/articles/figma-certifications"},
    ],
}


def recommend_certifications(
    missing_keywords: list[str],
    matched_keywords: list[str],
    max_missing: int = 5,
    max_strengthen: int = 2,
) -> list[CertificationSuggestion]:
    """Build a prioritized certification suggestion list.

    Priority 1: certifications for skills the job wants but the resume
    is missing (closes the biggest gaps first).
    Priority 2 (top-up only, capped small): certifications for skills
    already on the resume, framed as "get a credential to prove it".
    """
    suggestions: list[CertificationSuggestion] = []

    for skill in missing_keywords:
        options = CERT_MAP.get(skill)
        if not options:
            continue
        cert = options[0]
        suggestions.append(
            CertificationSuggestion(
                skill=skill,
                name=cert["name"],
                provider=cert["provider"],
                link=cert["link"],
                reason="missing",
            )
        )
        if len(suggestions) >= max_missing:
            break

    strengthen_count = 0
    if strengthen_count < max_strengthen:
        for skill in matched_keywords:
            if strengthen_count >= max_strengthen:
                break
            options = CERT_MAP.get(skill)
            if not options:
                continue
            cert = options[0]
            suggestions.append(
                CertificationSuggestion(
                    skill=skill,
                    name=cert["name"],
                    provider=cert["provider"],
                    link=cert["link"],
                    reason="strengthen",
                )
            )
            strengthen_count += 1

    return suggestions
