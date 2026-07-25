# sanity_check.py
from sim.oracle import Oracle
o = Oracle()
sample = """RESOLUTION: FR-008: Automated Review Report Generation

RATIONALE: The development team needs to ensure that the system generates review reports automatically...

Rewritten requirement: **FR-008: Automated Review Report Generation**

Description: The system shall automatically generate review reports based on the selected template.

CONFLICT 2: Missing non-functional requirements"""
print(o._extract_resolutions(sample))