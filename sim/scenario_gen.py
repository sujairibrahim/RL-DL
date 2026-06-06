"""
remarl/sim/scenario_gen.py
--------------------------
Synthetic Requirements Engineering scenario generator.

Each scenario is a complete "project brief" with:
  - rough_idea       : the vague input a stakeholder would give
  - domain           : the application domain
  - ground_truth_reqs: the correct functional requirements (oracle answer)
  - hidden_reqs      : subset the agents must DISCOVER through elicitation
  - nfr              : non-functional requirements
  - stakeholders     : list of stakeholder personas with interests
  - domain_entities  : key entities the modeler should extract
  - conflicts        : intentional contradictions to test the negotiator

Design principle:
  hidden_reqs are randomly withheld from the initial prompt.
  If the agents elicit well, they will surface them.
  The Oracle scores coverage against ground_truth_reqs.
  This gives a clean training signal without needing real projects.
"""

import json
import random
import pathlib
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────

@dataclass
class Stakeholder:
    name: str
    role: str
    primary_interest: str
    secondary_interest: str
    conflict_with: Optional[str] = None  # role this stakeholder conflicts with


@dataclass
class Scenario:
    scenario_id: str
    domain: str
    rough_idea: str
    ground_truth_reqs: List[str]
    hidden_reqs: List[str]          # subset of ground_truth that starts hidden
    visible_reqs: List[str]         # ground_truth minus hidden (shown in prompt)
    nfr: List[str]
    stakeholders: List[Stakeholder]
    domain_entities: List[str]
    conflicts: List[dict]           # [{"req_a": ..., "req_b": ..., "type": ...}]
    difficulty: str                 # "easy" | "medium" | "hard"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d["stakeholders"] = [Stakeholder(**s) for s in d["stakeholders"]]
        return cls(**d)


# ─────────────────────────────────────────────
#  Domain template library
#  30 domains across 6 sectors
# ─────────────────────────────────────────────

DOMAIN_TEMPLATES = [

    # ── SECTOR 1: E-COMMERCE & RETAIL ────────────────────────────────────

    {
        "domain": "e_commerce_marketplace",
        "rough_idea": "An online marketplace where sellers can list products and buyers can purchase them securely.",
        "ground_truth_reqs": [
            "The system shall allow sellers to register, create a store profile, and list products with images, descriptions, and prices.",
            "The system shall allow buyers to search for products by keyword, category, and price range.",
            "The system shall provide a shopping cart that persists across user sessions.",
            "The system shall process payments via credit card, debit card, and PayPal.",
            "The system shall send order confirmation and shipping update emails to buyers.",
            "The system shall allow buyers to leave ratings and reviews for completed purchases.",
            "The system shall provide sellers with a dashboard showing sales, revenue, and inventory.",
            "The system shall enforce a return and refund workflow with seller approval.",
        ],
        "nfr": [
            "Page load time shall not exceed 2 seconds under normal load.",
            "The system shall be available 99.9% of the time.",
            "All payment data shall be encrypted using AES-256.",
            "The system shall comply with PCI-DSS standards.",
        ],
        "stakeholders": [
            Stakeholder("Alice", "buyer", "low prices and fast delivery", "easy returns", "seller"),
            Stakeholder("Bob", "seller", "high visibility and low commission fees", "analytics", "buyer"),
            Stakeholder("Carol", "platform_admin", "fraud prevention and compliance", "revenue", None),
        ],
        "domain_entities": ["User", "Seller", "Product", "Cart", "Order", "Payment", "Review", "Category"],
        "conflicts": [
            {"req_a": "Buyers shall receive full refunds within 24 hours.",
             "req_b": "Sellers shall have 7 days to approve or deny refund requests.",
             "type": "temporal_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "subscription_box",
        "rough_idea": "A service that sends personalised monthly product boxes to subscribers based on their preferences.",
        "ground_truth_reqs": [
            "The system shall allow users to complete a preference questionnaire during onboarding.",
            "The system shall generate a personalised box selection based on user preferences and past ratings.",
            "The system shall support monthly, quarterly, and annual subscription plans.",
            "The system shall allow users to pause or cancel their subscription at any time.",
            "The system shall send a box preview notification 5 days before shipping.",
            "The system shall allow users to swap up to 2 products from their upcoming box.",
            "The system shall charge subscriptions automatically on the renewal date.",
            "The system shall track delivery status and notify users at each shipping milestone.",
        ],
        "nfr": [
            "Personalisation algorithm shall run within 500ms.",
            "System shall handle 10,000 concurrent active subscribers.",
            "GDPR compliance for user preference data.",
        ],
        "stakeholders": [
            Stakeholder("Dana", "subscriber", "personalised relevant products", "surprise factor", None),
            Stakeholder("Eve", "curator", "manageable curation workload", "product diversity", None),
            Stakeholder("Frank", "logistics_manager", "predictable fulfilment volumes", "on-time delivery", None),
        ],
        "domain_entities": ["Subscriber", "Box", "Product", "Subscription", "Preference", "Shipment"],
        "conflicts": [
            {"req_a": "Users shall be able to cancel with no notice period.",
             "req_b": "Boxes shall be prepared 10 days before the shipping date, making late cancellations non-refundable.",
             "type": "business_rule_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 2: HEALTHCARE ─────────────────────────────────────────────

    {
        "domain": "patient_portal",
        "rough_idea": "A web portal where patients can view their medical records, book appointments, and message their doctors.",
        "ground_truth_reqs": [
            "The system shall allow patients to register using their NHS/insurance number and verify their identity.",
            "The system shall display a patient's medical history, test results, and current prescriptions.",
            "The system shall allow patients to book, reschedule, and cancel appointments online.",
            "The system shall provide a secure messaging channel between patients and their assigned GP.",
            "The system shall allow doctors to update patient records and issue electronic prescriptions.",
            "The system shall send appointment reminders via SMS and email 24 hours before.",
            "The system shall allow patients to download their records as a PDF.",
            "The system shall enforce role-based access so patients cannot view other patients' records.",
        ],
        "nfr": [
            "System shall comply with HL7 FHIR standards for health data.",
            "All data at rest shall be encrypted using AES-256.",
            "System shall comply with HIPAA and GDPR.",
            "Authentication shall use two-factor authentication.",
            "System uptime shall be 99.95%.",
        ],
        "stakeholders": [
            Stakeholder("Grace", "patient", "easy access to own records", "privacy", "doctor"),
            Stakeholder("Henry", "gp_doctor", "efficient appointment management", "clinical accuracy", "patient"),
            Stakeholder("Irene", "hospital_admin", "regulatory compliance", "cost reduction", None),
        ],
        "domain_entities": ["Patient", "Doctor", "Appointment", "MedicalRecord", "Prescription", "Message", "TestResult"],
        "conflicts": [
            {"req_a": "Patients shall have immediate access to all test results.",
             "req_b": "Doctors shall review and annotate test results before patient release to prevent misinterpretation.",
             "type": "access_timing_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "mental_health_app",
        "rough_idea": "A mobile app for daily mood tracking, guided meditation, and connecting users with therapists.",
        "ground_truth_reqs": [
            "The system shall allow users to log their mood on a 1-10 scale with optional notes each day.",
            "The system shall display a mood trend chart over the past 30 days.",
            "The system shall provide a library of guided meditation sessions categorised by duration and goal.",
            "The system shall match users with licensed therapists based on their stated concerns.",
            "The system shall support video, voice, and text therapy sessions within the app.",
            "The system shall send a daily check-in notification at a user-configured time.",
            "The system shall detect 3 consecutive low-mood entries and prompt access to crisis resources.",
            "The system shall allow users to export their mood data for sharing with their therapist.",
        ],
        "nfr": [
            "All therapy session data shall be end-to-end encrypted.",
            "Crisis detection algorithm shall have false negative rate below 5%.",
            "App shall function with degraded connectivity for mood logging.",
            "HIPAA compliance mandatory.",
        ],
        "stakeholders": [
            Stakeholder("James", "end_user", "private mood tracking without stigma", "therapist access", None),
            Stakeholder("Karen", "therapist", "structured patient data before sessions", "session scheduling", None),
            Stakeholder("Leo", "clinical_director", "evidence-based intervention triggers", "liability management", None),
        ],
        "domain_entities": ["User", "MoodEntry", "Therapist", "Session", "MeditationTrack", "CrisisAlert"],
        "conflicts": [
            {"req_a": "User mood data shall never be shared without explicit consent.",
             "req_b": "In the event of crisis indicators, the system shall notify an emergency contact automatically.",
             "type": "privacy_safety_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 3: EDUCATION ──────────────────────────────────────────────

    {
        "domain": "online_learning_platform",
        "rough_idea": "An e-learning platform where instructors create courses and students learn at their own pace.",
        "ground_truth_reqs": [
            "The system shall allow instructors to create courses with video lectures, quizzes, and assignments.",
            "The system shall allow students to enrol in free and paid courses.",
            "The system shall track student progress and resume from the last watched position.",
            "The system shall generate a certificate of completion when a student passes all assessments.",
            "The system shall provide a discussion forum per course for student-instructor interaction.",
            "The system shall support multiple-choice, short answer, and coding exercise question types.",
            "The system shall allow instructors to set course prerequisites.",
            "The system shall provide instructors with analytics on student engagement and completion rates.",
        ],
        "nfr": [
            "Video streaming shall adapt to available bandwidth automatically.",
            "Platform shall support 50,000 concurrent learners.",
            "WCAG 2.1 AA accessibility compliance.",
            "System shall support English, Spanish, French, and Mandarin.",
        ],
        "stakeholders": [
            Stakeholder("Mia", "student", "self-paced affordable learning", "recognised certificates", None),
            Stakeholder("Noah", "instructor", "easy course creation tools", "revenue share", "student"),
            Stakeholder("Olivia", "platform_owner", "content quality and platform growth", "monetisation", None),
        ],
        "domain_entities": ["Student", "Instructor", "Course", "Lecture", "Quiz", "Certificate", "Forum", "Enrolment"],
        "conflicts": [
            {"req_a": "Instructors shall set their own pricing with no restrictions.",
             "req_b": "Platform shall enforce a maximum course price of £200 to maintain accessibility.",
             "type": "pricing_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "school_management_system",
        "rough_idea": "A system for schools to manage student records, timetables, attendance, and parent communication.",
        "ground_truth_reqs": [
            "The system shall maintain a complete academic record for each student including grades and attendance.",
            "The system shall generate class timetables based on teacher availability and room allocation.",
            "The system shall record daily attendance per class and flag absences to form teachers.",
            "The system shall allow teachers to submit grades for assignments and exams.",
            "The system shall generate progress reports each term and make them available to parents.",
            "The system shall provide a messaging system between teachers and parents.",
            "The system shall allow parents to submit absence notifications.",
            "The system shall track and report on SEND (special educational needs) student support plans.",
        ],
        "nfr": [
            "All student data shall comply with FERPA and GDPR.",
            "System shall be accessible on tablets used in classrooms.",
            "System shall support batch import of student data via CSV.",
        ],
        "stakeholders": [
            Stakeholder("Paul", "teacher", "minimal admin overhead", "clear grade tracking", None),
            Stakeholder("Quinn", "parent", "visibility of child's progress", "direct teacher communication", None),
            Stakeholder("Rachel", "headteacher", "whole-school analytics", "ofsted compliance", None),
        ],
        "domain_entities": ["Student", "Teacher", "Parent", "Class", "Timetable", "Attendance", "Grade", "Report"],
        "conflicts": [
            {"req_a": "Parents shall have real-time access to their child's grades.",
             "req_b": "Teachers shall have the ability to lock grades during moderation periods.",
             "type": "access_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 4: FINTECH & BANKING ──────────────────────────────────────

    {
        "domain": "personal_finance_app",
        "rough_idea": "A mobile app that connects to bank accounts and helps users track spending, set budgets, and save money.",
        "ground_truth_reqs": [
            "The system shall connect to user bank accounts via Open Banking API (PSD2 compliant).",
            "The system shall automatically categorise transactions into spending categories.",
            "The system shall allow users to set monthly budgets per category.",
            "The system shall alert users when they reach 80% of a budget limit.",
            "The system shall display net worth by aggregating all connected accounts.",
            "The system shall allow users to set savings goals with target amounts and dates.",
            "The system shall generate a monthly spending report with category breakdown.",
            "The system shall allow manual transaction entry for cash spending.",
        ],
        "nfr": [
            "Bank connection shall use read-only OAuth tokens — no write access.",
            "All financial data shall be encrypted at rest and in transit.",
            "FCA regulated and PSD2 compliant.",
            "App shall not store full bank credentials.",
        ],
        "stakeholders": [
            Stakeholder("Sam", "end_user", "clear spending insight", "financial privacy", None),
            Stakeholder("Tara", "product_manager", "engagement and retention", "monetisation via premium tier", None),
            Stakeholder("Uma", "compliance_officer", "FCA and PSD2 compliance", "data minimisation", None),
        ],
        "domain_entities": ["User", "BankAccount", "Transaction", "Budget", "Category", "SavingsGoal", "Report"],
        "conflicts": [
            {"req_a": "System shall retain transaction history indefinitely for trend analysis.",
             "req_b": "GDPR requires data to be deleted upon user request within 30 days.",
             "type": "retention_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "peer_lending_platform",
        "rough_idea": "A platform where individuals can lend money to small businesses and earn interest returns.",
        "ground_truth_reqs": [
            "The system shall allow borrowers to submit loan applications with business details and loan purpose.",
            "The system shall perform automated credit risk assessment on borrower applications.",
            "The system shall allow lenders to browse verified loan listings and commit funds.",
            "The system shall distribute borrower repayments proportionally to all contributing lenders.",
            "The system shall provide a secondary market where lenders can sell their loan parts.",
            "The system shall send monthly statements to lenders showing interest earned.",
            "The system shall enforce FCA-regulated investor limits for retail investors.",
            "The system shall place defaulted loans into a collections workflow automatically.",
        ],
        "nfr": [
            "Platform shall be FCA authorised and regulated.",
            "Credit scoring model shall be explainable (no black-box decisions).",
            "All fund transfers shall use ring-fenced client money accounts.",
            "System shall pass annual penetration tests.",
        ],
        "stakeholders": [
            Stakeholder("Victor", "lender", "attractive returns with managed risk", "liquidity via secondary market", "borrower"),
            Stakeholder("Wendy", "borrower", "fast approval and fair rates", "privacy of financials", "lender"),
            Stakeholder("Xavier", "risk_officer", "default rate below 3%", "regulatory compliance", None),
        ],
        "domain_entities": ["Lender", "Borrower", "LoanApplication", "LoanListing", "Investment", "Repayment", "SecondaryMarket"],
        "conflicts": [
            {"req_a": "Borrowers shall receive loan decisions within 24 hours.",
             "req_b": "Risk assessment shall include manual review for loans above £50,000.",
             "type": "speed_thoroughness_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 5: LOGISTICS & IoT ────────────────────────────────────────

    {
        "domain": "fleet_management",
        "rough_idea": "A system to track a company's delivery fleet in real time, optimise routes, and manage vehicle maintenance.",
        "ground_truth_reqs": [
            "The system shall display real-time GPS location of all vehicles on a map.",
            "The system shall calculate and suggest optimal delivery routes based on traffic data.",
            "The system shall send alerts when a vehicle deviates from its planned route.",
            "The system shall track fuel consumption per vehicle and flag inefficient driving behaviour.",
            "The system shall schedule preventive maintenance based on mileage and engine hours.",
            "The system shall allow drivers to log delivery confirmations via mobile app.",
            "The system shall generate end-of-day reports on deliveries completed, failed, and pending.",
            "The system shall integrate with the company's existing ERP for order data.",
        ],
        "nfr": [
            "GPS tracking shall update every 30 seconds.",
            "Mobile app shall work offline and sync when connectivity is restored.",
            "System shall handle a fleet of up to 500 vehicles.",
            "Driver data shall comply with GDPR.",
        ],
        "stakeholders": [
            Stakeholder("Yara", "fleet_manager", "full vehicle visibility", "cost reduction", None),
            Stakeholder("Zack", "driver", "simple mobile interface", "privacy of location outside working hours", "fleet_manager"),
            Stakeholder("Anna", "operations_director", "delivery SLA compliance", "ERP integration", None),
        ],
        "domain_entities": ["Vehicle", "Driver", "Route", "Delivery", "MaintenanceSchedule", "FuelLog", "Alert"],
        "conflicts": [
            {"req_a": "System shall track driver location continuously during working hours.",
             "req_b": "Drivers shall have the ability to disable location tracking during breaks.",
             "type": "surveillance_privacy_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "smart_building_iot",
        "rough_idea": "An IoT platform to manage energy, security, and comfort systems in a commercial office building.",
        "ground_truth_reqs": [
            "The system shall integrate with HVAC, lighting, and access control sensors via MQTT.",
            "The system shall automatically adjust temperature based on occupancy detected by motion sensors.",
            "The system shall allow building managers to set energy budgets and alert when exceeded.",
            "The system shall provide a floor-plan view showing real-time occupancy per zone.",
            "The system shall control access door locks and log all entry and exit events.",
            "The system shall detect anomalies such as unusual after-hours access and send alerts.",
            "The system shall generate monthly energy consumption reports by floor and system type.",
            "The system shall allow remote control of any connected device from the management dashboard.",
        ],
        "nfr": [
            "Device command latency shall be under 500ms.",
            "System shall support 10,000 concurrent IoT device connections.",
            "All device communications shall use TLS 1.3.",
            "System shall remain operational if internet connectivity is lost (local fallback mode).",
        ],
        "stakeholders": [
            Stakeholder("Ben", "building_manager", "energy cost reduction", "tenant comfort", None),
            Stakeholder("Cleo", "security_officer", "access control audit trail", "intrusion detection", None),
            Stakeholder("Dan", "tenant_company", "comfortable working environment", "privacy from landlord", "building_manager"),
        ],
        "domain_entities": ["Sensor", "Device", "Zone", "OccupancyEvent", "EnergyReading", "AccessEvent", "Alert"],
        "conflicts": [
            {"req_a": "Building manager shall have access to individual occupancy data per desk.",
             "req_b": "Tenant privacy policy prohibits individual-level location tracking of employees.",
             "type": "granularity_privacy_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 6: SOCIAL & PRODUCTIVITY ─────────────────────────────────

    {
        "domain": "task_management_saas",
        "rough_idea": "A project management tool for software teams to track tasks, sprints, and bugs.",
        "ground_truth_reqs": [
            "The system shall allow users to create projects and invite team members with defined roles.",
            "The system shall support Kanban boards and sprint-based backlog management.",
            "The system shall allow tasks to be assigned to team members with due dates and priority levels.",
            "The system shall track time logged against each task.",
            "The system shall generate velocity charts and burndown charts for each sprint.",
            "The system shall provide a GitHub integration to link commits and PRs to tasks.",
            "The system shall send daily digest emails with overdue and at-risk tasks.",
            "The system shall support custom workflow states beyond the default To Do / In Progress / Done.",
        ],
        "nfr": [
            "API shall respond within 200ms at the 99th percentile.",
            "Data shall be exportable in CSV and JSON formats.",
            "System shall support SSO via SAML 2.0 for enterprise customers.",
            "99.9% monthly uptime SLA.",
        ],
        "stakeholders": [
            Stakeholder("Elle", "developer", "minimal process overhead", "clear task ownership", None),
            Stakeholder("Fred", "product_manager", "sprint planning visibility", "stakeholder reporting", None),
            Stakeholder("Gina", "engineering_manager", "team velocity tracking", "resource allocation", None),
        ],
        "domain_entities": ["Project", "Task", "Sprint", "User", "Team", "TimeLog", "GitCommit", "WorkflowState"],
        "conflicts": [
            {"req_a": "Developers shall be able to re-estimate tasks at any time during a sprint.",
             "req_b": "Sprint commitments shall be locked at sprint start to maintain predictability.",
             "type": "agile_methodology_conflict"},
        ],
        "difficulty": "easy",
    },

    {
        "domain": "remote_team_collaboration",
        "rough_idea": "A virtual office platform where distributed teams can have video calls, share screens, and maintain persistent chat rooms.",
        "ground_truth_reqs": [
            "The system shall provide persistent text channels organised by team and topic.",
            "The system shall support video calls for up to 50 participants with screen sharing.",
            "The system shall allow users to set a status indicating availability.",
            "The system shall provide a shared virtual whiteboard for brainstorming sessions.",
            "The system shall allow file sharing up to 500MB per file with preview support.",
            "The system shall record meetings on request and make recordings available for 30 days.",
            "The system shall support threaded replies within channels to keep conversations organised.",
            "The system shall provide a company-wide searchable message archive.",
        ],
        "nfr": [
            "Video calls shall maintain quality at 720p on a 5Mbps connection.",
            "Message delivery latency shall be under 100ms.",
            "End-to-end encryption for direct messages.",
            "GDPR compliance for EU users.",
        ],
        "stakeholders": [
            Stakeholder("Hugo", "remote_employee", "seamless async communication", "work-life boundary", None),
            Stakeholder("Isla", "hr_manager", "culture and engagement monitoring", "compliance", None),
            Stakeholder("Jake", "cto", "security and enterprise features", "integration with existing tools", None),
        ],
        "domain_entities": ["User", "Channel", "Message", "VideoCall", "Recording", "File", "Whiteboard"],
        "conflicts": [
            {"req_a": "HR shall have access to all message history for compliance purposes.",
             "req_b": "Direct messages shall be end-to-end encrypted and inaccessible to employers.",
             "type": "compliance_privacy_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 7: GOVERNMENT & PUBLIC ────────────────────────────────────

    {
        "domain": "citizen_services_portal",
        "rough_idea": "A government portal where citizens can apply for permits, pay taxes, and access public services online.",
        "ground_truth_reqs": [
            "The system shall allow citizens to create a verified digital identity using government ID.",
            "The system shall list all available services with eligibility criteria and required documents.",
            "The system shall allow citizens to submit permit applications and upload supporting documents.",
            "The system shall track application status and notify citizens at each stage.",
            "The system shall allow online payment of council tax, fines, and fees.",
            "The system shall provide a live chat and callback request option for citizen support.",
            "The system shall generate official PDF acknowledgements for every submitted application.",
            "The system shall support Welsh and English languages for Wales-based deployments.",
        ],
        "nfr": [
            "WCAG 2.2 AA accessibility compliance mandatory.",
            "System shall comply with UK Government Digital Service (GDS) standards.",
            "99.99% uptime during business hours.",
            "All data stored in UK data centres.",
        ],
        "stakeholders": [
            Stakeholder("Lena", "citizen", "quick self-service without queuing", "privacy of personal data", None),
            Stakeholder("Mike", "council_caseworker", "structured application data", "manageable caseload", None),
            Stakeholder("Nina", "digital_director", "GDS compliance and cost savings", "accessibility", None),
        ],
        "domain_entities": ["Citizen", "Application", "Service", "Document", "Payment", "CaseWorker", "Notification"],
        "conflicts": [
            {"req_a": "System shall auto-approve low-risk permit applications within 24 hours.",
             "req_b": "All permit approvals shall require manual sign-off by a qualified caseworker.",
             "type": "automation_accountability_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 8: ENTERTAINMENT & MEDIA ──────────────────────────────────

    {
        "domain": "video_streaming",
        "rough_idea": "A video streaming platform for watching movies and TV shows on any device.",
        "ground_truth_reqs": [
            "The system shall provide a content catalogue with search, genre filter, and recommendation.",
            "The system shall stream video in 480p, 720p, 1080p, and 4K based on network speed.",
            "The system shall allow users to download content for offline viewing.",
            "The system shall support user profiles within a single household account.",
            "The system shall track watch history and resume playback from the last position.",
            "The system shall implement parental controls with PIN-protected content ratings.",
            "The system shall provide subtitle support in at least 10 languages.",
            "The system shall support simultaneous streaming on up to 4 devices per account.",
        ],
        "nfr": [
            "Video shall start within 3 seconds on a 10Mbps connection.",
            "DRM protection via Widevine or FairPlay.",
            "CDN shall serve content from the nearest edge node.",
            "System shall handle 1 million concurrent streams.",
        ],
        "stakeholders": [
            Stakeholder("Ora", "subscriber", "wide content library and offline access", "affordable pricing", None),
            Stakeholder("Pete", "content_partner", "DRM enforcement and royalty tracking", "audience analytics", None),
            Stakeholder("Rita", "product_manager", "subscriber growth and retention", "personalisation accuracy", None),
        ],
        "domain_entities": ["Content", "User", "Profile", "WatchHistory", "Download", "Subscription", "Recommendation"],
        "conflicts": [
            {"req_a": "Downloads shall be available for unlimited offline viewing.",
             "req_b": "Content licences restrict offline viewing to 30 days and 5 saves per title.",
             "type": "licence_constraint_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 1 CONTINUED: E-COMMERCE & RETAIL ──────────────────────────

    {
        "domain": "food_delivery_platform",
        "rough_idea": "A platform connecting customers with local restaurants for food ordering and doorstep delivery.",
        "ground_truth_reqs": [
            "The system shall allow customers to browse restaurants by cuisine type, rating, and delivery time.",
            "The system shall allow customers to place orders and customise items with add-ons or removals.",
            "The system shall calculate delivery fees and estimated arrival times based on distance and demand.",
            "The system shall assign delivery orders to the nearest available driver automatically.",
            "The system shall allow customers to track their order in real time on a map.",
            "The system shall support payment via card, wallet, and cash on delivery.",
            "The system shall send push notifications at order confirmation, preparation, and delivery stages.",
            "The system shall allow customers to rate drivers and restaurants after each order.",
        ],
        "nfr": [
            "Order assignment to driver shall occur within 30 seconds.",
            "Real-time tracking shall update every 10 seconds.",
            "System shall handle 5,000 concurrent orders.",
            "Mobile app shall function on Android 8.0+ and iOS 14+.",
        ],
        "stakeholders": [
            Stakeholder("Amy", "customer", "fast delivery and accurate orders", "variety of restaurants", "restaurant"),
            Stakeholder("Bruno", "restaurant_owner", "accurate order relay and fair commission", "menu control", "customer"),
            Stakeholder("Carlos", "delivery_driver", "optimised routes and fair pay", "flexible working hours", None),
        ],
        "domain_entities": ["Customer", "Restaurant", "Order", "Driver", "MenuItem", "Delivery", "Payment", "Rating"],
        "conflicts": [
            {"req_a": "Restaurants shall have up to 10 minutes to confirm an order before it is auto-cancelled.",
             "req_b": "Customers shall receive a delivery guarantee within 30 minutes of ordering.",
             "type": "timing_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "auction_platform",
        "rough_idea": "An online auction site where sellers list items and buyers bid in real time to win them.",
        "ground_truth_reqs": [
            "The system shall allow sellers to list items with photos, descriptions, starting bid, and auction end time.",
            "The system shall allow buyers to place bids and show the current highest bid in real time.",
            "The system shall support proxy bidding where the system auto-bids up to a user-set maximum.",
            "The system shall notify all bidders when they are outbid via email and push notification.",
            "The system shall declare the highest bidder the winner and initiate payment automatically.",
            "The system shall hold payment in escrow until the buyer confirms item receipt.",
            "The system shall allow sellers to set a reserve price that must be met for the sale to proceed.",
            "The system shall flag and suspend accounts with repeated non-payment or fraudulent listings.",
        ],
        "nfr": [
            "Bid updates shall be reflected across all clients within 500ms.",
            "System shall support 10,000 simultaneous live auctions.",
            "All payment data shall comply with PCI-DSS.",
            "Auction end-time precision shall be accurate to within 1 second.",
        ],
        "stakeholders": [
            Stakeholder("Diana", "seller", "maximum sale price and low fees", "buyer trust", "buyer"),
            Stakeholder("Ethan", "buyer", "fair bidding and secure payment", "item authenticity", "seller"),
            Stakeholder("Fiona", "platform_operator", "fraud prevention and revenue", "dispute resolution", None),
        ],
        "domain_entities": ["Seller", "Buyer", "Listing", "Bid", "Auction", "Escrow", "Payment", "Dispute"],
        "conflicts": [
            {"req_a": "Sellers shall be able to cancel a listing at any time before auction ends.",
             "req_b": "Once a bid is placed, the listing shall be binding and cannot be cancelled.",
             "type": "commitment_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "rental_marketplace",
        "rough_idea": "A peer-to-peer platform where owners can rent out equipment, tools, or spaces to local renters.",
        "ground_truth_reqs": [
            "The system shall allow owners to list rentable items with availability calendar, daily rate, and deposit.",
            "The system shall allow renters to search listings by category, location, and date range.",
            "The system shall provide an availability calendar that prevents double bookings.",
            "The system shall process rental payments and hold a refundable deposit in escrow.",
            "The system shall require renters to submit a photo of the item on collection and return.",
            "The system shall release the deposit to the owner if damage is reported within 48 hours.",
            "The system shall allow both parties to leave reviews after each rental period.",
            "The system shall provide an in-app messaging system for owner-renter coordination.",
        ],
        "nfr": [
            "Calendar availability shall update in real time to prevent conflicts.",
            "Damage photo uploads shall support images up to 20MB.",
            "System shall comply with GDPR for user personal data.",
            "Payment processing shall settle within 2 business days.",
        ],
        "stakeholders": [
            Stakeholder("George", "owner", "maximum rental income and item protection", "easy listing management", "renter"),
            Stakeholder("Hannah", "renter", "affordable access to equipment", "transparent pricing", "owner"),
            Stakeholder("Ivan", "platform_manager", "dispute resolution and trust", "insurance partnerships", None),
        ],
        "domain_entities": ["Owner", "Renter", "Listing", "Booking", "Payment", "Deposit", "Review", "DamageReport"],
        "conflicts": [
            {"req_a": "Renters shall receive a full deposit refund if returned on time with no damage.",
             "req_b": "Owners shall have 72 hours to inspect items after return before deposit is released.",
             "type": "timing_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 2 CONTINUED: HEALTHCARE ───────────────────────────────────

    {
        "domain": "telemedicine_platform",
        "rough_idea": "A platform enabling patients to consult with licensed doctors via video, voice, or chat from home.",
        "ground_truth_reqs": [
            "The system shall allow patients to book on-demand or scheduled consultations with licensed doctors.",
            "The system shall match patients with available doctors based on specialty and language.",
            "The system shall provide HIPAA-compliant video and voice consultation rooms.",
            "The system shall allow doctors to issue digital prescriptions that can be sent to pharmacies.",
            "The system shall maintain a consultation history and summary accessible to both patient and doctor.",
            "The system shall support integration with wearable device data for remote monitoring.",
            "The system shall allow follow-up messaging between patient and doctor for 48 hours post-consultation.",
            "The system shall process insurance claims and direct billing for covered consultations.",
        ],
        "nfr": [
            "Video consultation quality shall maintain 720p at 1Mbps upload.",
            "System shall comply with HIPAA, GDPR, and local telehealth regulations.",
            "Average wait time for on-demand consultations shall be under 10 minutes.",
            "All health data shall be encrypted end-to-end.",
        ],
        "stakeholders": [
            Stakeholder("Julia", "patient", "quick access to quality medical advice", "data privacy", "doctor"),
            Stakeholder("Kevin", "doctor", "efficient patient management and accurate records", "liability protection", "patient"),
            Stakeholder("Laura", "insurance_provider", "claim verification and fraud prevention", "cost control", None),
        ],
        "domain_entities": ["Patient", "Doctor", "Consultation", "Prescription", "InsuranceClaim", "WearableData", "Message"],
        "conflicts": [
            {"req_a": "Patients shall have immediate access to their full consultation notes.",
             "req_b": "Doctors shall retain the right to annotate and finalise notes within 24 hours before release.",
             "type": "access_timing_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "pharmacy_management",
        "rough_idea": "A system for pharmacies to manage inventory, process prescriptions, and serve customers efficiently.",
        "ground_truth_reqs": [
            "The system shall receive and validate electronic prescriptions from authorised healthcare providers.",
            "The system shall check drug interactions and allergies before dispensing medication.",
            "The system shall track medication inventory and trigger low-stock alerts automatically.",
            "The system shall allow customers to request repeat prescriptions via the patient portal.",
            "The system shall record all dispensed medications and maintain a full audit trail.",
            "The system shall support home delivery scheduling for eligible prescriptions.",
            "The system shall integrate with the national prescription database for validation.",
            "The system shall generate compliance reports for regulatory inspections.",
        ],
        "nfr": [
            "Drug interaction checks shall complete within 2 seconds.",
            "System shall comply with NHS and MHRA digital health standards.",
            "Audit logs shall be retained for a minimum of 10 years.",
            "System shall achieve 99.9% uptime during pharmacy opening hours.",
        ],
        "stakeholders": [
            Stakeholder("Mark", "pharmacist", "accurate dispensing and error prevention", "efficient workflow", None),
            Stakeholder("Nancy", "patient", "fast prescription fulfilment and home delivery", "medication privacy", None),
            Stakeholder("Oscar", "pharmacy_manager", "stock optimisation and regulatory compliance", "revenue", None),
        ],
        "domain_entities": ["Prescription", "Medication", "Patient", "Pharmacist", "Inventory", "Dispensing", "Supplier", "AuditLog"],
        "conflicts": [
            {"req_a": "The system shall automatically dispense repeat prescriptions without pharmacist review.",
             "req_b": "Regulations require pharmacist sign-off on every dispensed medication.",
             "type": "automation_compliance_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "clinical_trial_management",
        "rough_idea": "A platform to manage clinical drug trials including participant recruitment, data collection, and regulatory reporting.",
        "ground_truth_reqs": [
            "The system shall manage trial protocols and track amendments with full version history.",
            "The system shall support participant screening, enrolment, and randomisation workflows.",
            "The system shall collect adverse event reports and escalate serious events within 24 hours.",
            "The system shall enforce data entry validation rules defined in the trial protocol.",
            "The system shall provide role-based access for investigators, monitors, and sponsors.",
            "The system shall generate 21 CFR Part 11 compliant electronic signatures for all critical actions.",
            "The system shall integrate with laboratory information systems for biomarker data import.",
            "The system shall produce CDISC-compliant data exports for regulatory submissions.",
        ],
        "nfr": [
            "System shall comply with ICH E6 GCP guidelines.",
            "All data modifications shall be logged with timestamp, user, and reason.",
            "System shall support trials across 50 countries with localised date formats.",
            "Data exports shall be available within 4 hours of request.",
        ],
        "stakeholders": [
            Stakeholder("Peter", "principal_investigator", "protocol compliance and participant safety", "data quality", None),
            Stakeholder("Quinn", "sponsor", "on-time regulatory submissions", "cost control", "investigator"),
            Stakeholder("Rosa", "regulator", "GCP compliance and participant protection", "audit trail completeness", None),
        ],
        "domain_entities": ["Trial", "Participant", "Protocol", "AdverseEvent", "DataPoint", "Investigator", "Site", "Submission"],
        "conflicts": [
            {"req_a": "Sponsors shall have real-time access to unblinded trial data for safety monitoring.",
             "req_b": "Trial blinding shall be maintained until the pre-specified unblinding event to prevent bias.",
             "type": "safety_integrity_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 3 CONTINUED: EDUCATION ────────────────────────────────────

    {
        "domain": "university_admission_system",
        "rough_idea": "A system for universities to manage undergraduate applications, reviews, and admissions decisions.",
        "ground_truth_reqs": [
            "The system shall allow applicants to create accounts and submit applications with personal statements.",
            "The system shall validate that all required documents are uploaded before submission.",
            "The system shall route applications to appropriate admissions reviewers based on programme.",
            "The system shall support committee scoring rubrics with configurable criteria and weightings.",
            "The system shall generate ranked shortlists and offer letters automatically based on scores.",
            "The system shall allow applicants to accept, decline, or defer offers with a deadline.",
            "The system shall maintain a complete audit log of all reviewer comments and decisions.",
            "The system shall produce equal opportunity monitoring reports for compliance.",
        ],
        "nfr": [
            "System shall handle 100,000 applications per cycle.",
            "Document uploads shall support PDF, Word, and image formats up to 10MB.",
            "System shall comply with GDPR and UK Equality Act 2010.",
            "Application portal shall be WCAG 2.1 AA compliant.",
        ],
        "stakeholders": [
            Stakeholder("Sara", "applicant", "clear requirements and fair assessment", "application status visibility", None),
            Stakeholder("Tom", "admissions_officer", "structured review workflow", "consistent criteria", "applicant"),
            Stakeholder("Uma", "compliance_manager", "equal opportunity compliance", "data retention", None),
        ],
        "domain_entities": ["Applicant", "Application", "Programme", "Reviewer", "Document", "Decision", "Offer", "Report"],
        "conflicts": [
            {"req_a": "Admissions decisions shall be made solely on academic merit scores.",
             "req_b": "Contextual admissions policy requires deprivation indicators to adjust offer thresholds.",
             "type": "policy_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "corporate_training_platform",
        "rough_idea": "An LMS for companies to deliver mandatory compliance training and professional development courses to employees.",
        "ground_truth_reqs": [
            "The system shall allow HR to assign mandatory training courses to employees with completion deadlines.",
            "The system shall send automated reminders to employees with overdue training.",
            "The system shall record training completion with timestamp and assessment score.",
            "The system shall generate compliance reports showing completion rates by department and course.",
            "The system shall support SCORM 1.2 and xAPI content standards for third-party courseware.",
            "The system shall allow managers to view their team's training status dashboard.",
            "The system shall issue digital certificates upon course completion.",
            "The system shall support blended learning with in-person session scheduling alongside online modules.",
        ],
        "nfr": [
            "Platform shall support SSO integration via SAML 2.0 and OIDC.",
            "Video modules shall support adaptive bitrate streaming.",
            "System shall handle 10,000 concurrent learners.",
            "GDPR compliance for employee learning data.",
        ],
        "stakeholders": [
            Stakeholder("Victor", "employee", "relevant engaging training with minimal interruption", "recognised certification", None),
            Stakeholder("Wendy", "hr_manager", "full compliance visibility and report generation", "deadline enforcement", None),
            Stakeholder("Xena", "l_and_d_manager", "course quality and learner engagement metrics", "content management", None),
        ],
        "domain_entities": ["Employee", "Course", "Module", "Enrolment", "Assessment", "Certificate", "Department", "Report"],
        "conflicts": [
            {"req_a": "Employees shall complete mandatory training within 30 days of assignment.",
             "req_b": "Managers shall be able to grant unlimited extensions to prevent work disruption.",
             "type": "compliance_flexibility_conflict"},
        ],
        "difficulty": "easy",
    },

    {
        "domain": "language_learning_app",
        "rough_idea": "A gamified mobile app to help users learn new languages through daily lessons, speaking exercises, and challenges.",
        "ground_truth_reqs": [
            "The system shall provide structured learning paths per language from beginner to advanced.",
            "The system shall deliver daily bite-sized lessons with vocabulary, grammar, and listening exercises.",
            "The system shall use speech recognition to assess pronunciation and provide feedback.",
            "The system shall implement a spaced repetition algorithm to optimise vocabulary review.",
            "The system shall award experience points, streaks, and badges to encourage daily engagement.",
            "The system shall allow users to join language leagues and compete on weekly leaderboards.",
            "The system shall provide AI-powered conversational practice simulating real dialogue.",
            "The system shall allow users to set daily learning goals and track time spent.",
        ],
        "nfr": [
            "Speech recognition shall achieve 85% accuracy for non-native speakers.",
            "App shall work offline for downloaded lessons.",
            "Daily lesson load time shall be under 1 second.",
            "COPPA compliance for users under 13.",
        ],
        "stakeholders": [
            Stakeholder("Yuki", "learner", "measurable progress and fun experience", "daily habit formation", None),
            Stakeholder("Zara", "content_linguist", "pedagogically sound lesson design", "language accuracy", None),
            Stakeholder("Alex", "product_manager", "daily active user retention", "premium conversion", None),
        ],
        "domain_entities": ["User", "Language", "Lesson", "Exercise", "VocabularyItem", "Progress", "Badge", "League"],
        "conflicts": [
            {"req_a": "The system shall show full leaderboard rankings to drive competition.",
             "req_b": "Users shall be able to opt out of leaderboards to reduce anxiety.",
             "type": "engagement_wellbeing_conflict"},
        ],
        "difficulty": "easy",
    },

    # ── SECTOR 4 CONTINUED: FINTECH & BANKING ────────────────────────────

    {
        "domain": "digital_banking_app",
        "rough_idea": "A fully digital bank with current accounts, savings, and instant payments via a mobile app.",
        "ground_truth_reqs": [
            "The system shall allow customers to open a current account with digital identity verification.",
            "The system shall support instant domestic and international bank transfers.",
            "The system shall provide virtual and physical debit card management within the app.",
            "The system shall allow customers to set spending limits and freeze or unfreeze their card instantly.",
            "The system shall provide real-time transaction notifications via push message.",
            "The system shall allow customers to create multiple savings pots with target amounts.",
            "The system shall offer a credit score tracking feature via Open Banking data.",
            "The system shall support biometric authentication for login and payment authorisation.",
        ],
        "nfr": [
            "Domestic transfers shall settle within 10 seconds via Faster Payments.",
            "System shall comply with FCA, PSD2, and AML regulations.",
            "App shall achieve 99.99% uptime with no scheduled maintenance windows.",
            "Biometric data shall never leave the user's device.",
        ],
        "stakeholders": [
            Stakeholder("Ben", "customer", "seamless mobile banking experience", "financial security", None),
            Stakeholder("Celia", "compliance_officer", "AML and KYC compliance", "fraud prevention", None),
            Stakeholder("Derek", "cto", "scalable architecture and security", "open banking integrations", None),
        ],
        "domain_entities": ["Customer", "Account", "Transaction", "Card", "SavingsPot", "Payment", "Notification", "CreditScore"],
        "conflicts": [
            {"req_a": "Account opening shall be completed in under 5 minutes with minimal documentation.",
             "req_b": "KYC regulations require enhanced due diligence for high-risk customers with full document review.",
             "type": "speed_compliance_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "insurance_claims_platform",
        "rough_idea": "A digital platform for customers to file insurance claims and for assessors to process and settle them.",
        "ground_truth_reqs": [
            "The system shall allow policyholders to file claims online by selecting cover type and describing the incident.",
            "The system shall allow claimants to upload photos, videos, and supporting documents.",
            "The system shall perform automated triage to classify claims by type, complexity, and fraud risk.",
            "The system shall assign low-complexity claims for automated settlement and complex claims to assessors.",
            "The system shall allow assessors to request additional information and set response deadlines.",
            "The system shall send real-time status updates to claimants at each stage of processing.",
            "The system shall process approved claim payments within 48 hours via bank transfer.",
            "The system shall flag suspicious claims for special investigation unit review.",
        ],
        "nfr": [
            "Automated claim triage shall complete within 60 seconds of submission.",
            "Fraud detection model shall achieve precision above 90%.",
            "System shall comply with FCA insurance regulations.",
            "All claim documents shall be retained for 7 years.",
        ],
        "stakeholders": [
            Stakeholder("Emma", "claimant", "fast fair settlement and clear communication", "minimal paperwork", "assessor"),
            Stakeholder("Felix", "claims_assessor", "complete accurate claim information", "manageable caseload", "claimant"),
            Stakeholder("Grace", "fraud_analyst", "early fraud detection", "minimal false positives", None),
        ],
        "domain_entities": ["Claimant", "Claim", "Policy", "Document", "Assessor", "Payment", "FraudFlag", "Notification"],
        "conflicts": [
            {"req_a": "Simple claims shall be auto-approved and paid within 24 hours.",
             "req_b": "All claims above £500 shall require manual assessor review before payment.",
             "type": "automation_oversight_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "crypto_exchange",
        "rough_idea": "A cryptocurrency trading platform for buying, selling, and storing digital assets securely.",
        "ground_truth_reqs": [
            "The system shall allow users to register and complete KYC and AML identity verification.",
            "The system shall support trading of at least 50 cryptocurrency pairs.",
            "The system shall provide a real-time order book with market and limit order types.",
            "The system shall execute matched trades within 100 milliseconds.",
            "The system shall maintain hot and cold wallet infrastructure with 95% of funds in cold storage.",
            "The system shall allow users to set up two-factor authentication for all withdrawals.",
            "The system shall provide a portfolio overview showing holdings, P&L, and transaction history.",
            "The system shall generate tax reports showing capital gains per jurisdiction.",
        ],
        "nfr": [
            "Trading engine shall process 10,000 transactions per second.",
            "System shall comply with FATF travel rule and local crypto regulations.",
            "Cold wallet transfers shall require multi-signature approval.",
            "Security audits shall be conducted quarterly.",
        ],
        "stakeholders": [
            Stakeholder("Harry", "retail_trader", "low fees and fast execution", "fund security", None),
            Stakeholder("Irene", "compliance_officer", "AML and KYC compliance", "regulatory reporting", None),
            Stakeholder("Jack", "security_engineer", "exchange security and cold storage", "penetration testing", None),
        ],
        "domain_entities": ["User", "Wallet", "Order", "Trade", "Currency", "Portfolio", "Transaction", "AuditLog"],
        "conflicts": [
            {"req_a": "Withdrawals shall be processed instantly to improve user experience.",
             "req_b": "Large withdrawals shall be delayed 24 hours for manual security review.",
             "type": "ux_security_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 5 CONTINUED: LOGISTICS & IoT ─────────────────────────────

    {
        "domain": "warehouse_management",
        "rough_idea": "A system for managing inventory, picking, packing, and dispatching orders in a distribution warehouse.",
        "ground_truth_reqs": [
            "The system shall track inventory location at bin level using barcode or RFID scanning.",
            "The system shall generate optimised pick lists for warehouse staff based on order priority.",
            "The system shall support wave picking to batch multiple orders for efficiency.",
            "The system shall guide pickers via mobile devices with step-by-step bin navigation.",
            "The system shall verify picked items against order lines to prevent mis-picks.",
            "The system shall integrate with carrier APIs to generate shipping labels on packing.",
            "The system shall update inventory in real time as goods are received and dispatched.",
            "The system shall generate KPI dashboards showing throughput, accuracy, and labour productivity.",
        ],
        "nfr": [
            "Barcode scan response shall be under 200ms.",
            "System shall handle 50,000 SKUs and 10,000 orders per day.",
            "Mobile devices shall operate reliably on warehouse Wi-Fi with intermittent coverage.",
            "System shall integrate with SAP and Oracle ERP via REST API.",
        ],
        "stakeholders": [
            Stakeholder("Kim", "warehouse_manager", "throughput and accuracy", "labour efficiency", None),
            Stakeholder("Leo", "picker", "clear simple instructions and minimal walking", "ergonomic device", None),
            Stakeholder("Mia", "operations_director", "SLA compliance and cost reduction", "ERP integration", None),
        ],
        "domain_entities": ["SKU", "Bin", "Order", "PickList", "Shipment", "Carrier", "Inventory", "Worker"],
        "conflicts": [
            {"req_a": "Pickers shall confirm each scan individually for maximum accuracy.",
             "req_b": "Scan confirmation steps slow throughput below KPI targets during peak periods.",
             "type": "accuracy_speed_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "cold_chain_monitoring",
        "rough_idea": "An IoT system to monitor temperature and humidity of pharmaceutical or food shipments throughout the supply chain.",
        "ground_truth_reqs": [
            "The system shall collect temperature and humidity readings from IoT sensors every 5 minutes.",
            "The system shall trigger alerts within 60 seconds when readings exceed defined thresholds.",
            "The system shall log all sensor readings with GPS location and timestamp.",
            "The system shall support configurable alert thresholds per product type or shipment.",
            "The system shall generate excursion reports automatically when threshold breaches occur.",
            "The system shall provide a chain-of-custody dashboard showing shipment status at each handover.",
            "The system shall integrate with logistics carrier APIs for shipment tracking.",
            "The system shall produce compliance reports suitable for FDA 21 CFR Part 211 audits.",
        ],
        "nfr": [
            "Sensor readings shall be transmitted even with cellular connectivity gaps via store-and-forward.",
            "Alert delivery shall achieve 99.9% reliability.",
            "Data retention shall meet a minimum of 7 years for pharmaceutical products.",
            "System shall support 100,000 concurrent active shipments.",
        ],
        "stakeholders": [
            Stakeholder("Nina", "logistics_manager", "full shipment visibility and compliance", "cost reduction", None),
            Stakeholder("Omar", "quality_manager", "excursion detection and product integrity", "audit readiness", None),
            Stakeholder("Paula", "regulatory_affairs", "FDA compliance and documentation", "audit trail", None),
        ],
        "domain_entities": ["Sensor", "Shipment", "Reading", "Alert", "Excursion", "Product", "Carrier", "ComplianceReport"],
        "conflicts": [
            {"req_a": "Cost-reduction targets require reducing sensor transmission frequency to every 30 minutes.",
             "req_b": "Pharmaceutical SOP requires temperature readings at minimum every 5 minutes during transit.",
             "type": "cost_compliance_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "smart_parking_system",
        "rough_idea": "An IoT-enabled system to guide drivers to available parking spaces and automate payment in city car parks.",
        "ground_truth_reqs": [
            "The system shall detect available parking spaces using in-ground sensors or cameras.",
            "The system shall display real-time space availability on entry signs and a mobile app.",
            "The system shall allow drivers to reserve a parking space in advance via the app.",
            "The system shall record entry and exit times using ANPR number plate recognition.",
            "The system shall calculate parking charges based on duration and tariff zone.",
            "The system shall support payment via app, contactless card, and mobile wallet.",
            "The system shall allow permit holders to access reserved zones without payment.",
            "The system shall generate occupancy analytics reports for city planning.",
        ],
        "nfr": [
            "Space availability data shall update within 10 seconds of a vehicle entering or leaving.",
            "ANPR recognition accuracy shall exceed 98%.",
            "Mobile app shall function on 3G networks.",
            "System shall comply with GDPR for vehicle registration data.",
        ],
        "stakeholders": [
            Stakeholder("Quinn", "driver", "quick space location and easy payment", "fair pricing", None),
            Stakeholder("Rachel", "parking_operator", "revenue maximisation and enforcement", "operational efficiency", "driver"),
            Stakeholder("Steve", "city_council", "traffic congestion reduction", "disabled bay compliance", None),
        ],
        "domain_entities": ["ParkingSpace", "Vehicle", "Reservation", "Payment", "PermitHolder", "Sensor", "Zone", "Report"],
        "conflicts": [
            {"req_a": "Reservations shall be held for 15 minutes after the scheduled arrival time.",
             "req_b": "Unreserved spaces during peak hours generate more revenue than held reservations.",
             "type": "ux_revenue_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 6 CONTINUED: SOCIAL & PRODUCTIVITY ────────────────────────

    {
        "domain": "social_media_platform",
        "rough_idea": "A social network where users post content, follow others, and engage through likes, comments, and shares.",
        "ground_truth_reqs": [
            "The system shall allow users to create profiles with bio, avatar, and privacy settings.",
            "The system shall allow users to create text, image, and short video posts.",
            "The system shall generate a personalised feed based on followed accounts and engagement history.",
            "The system shall allow users to like, comment on, and share posts from others.",
            "The system shall implement a reporting system for harmful content with moderator review.",
            "The system shall notify users of interactions with their content in real time.",
            "The system shall allow users to send direct messages to followers.",
            "The system shall enforce content policies by automatically detecting and flagging violating content.",
        ],
        "nfr": [
            "Feed generation shall complete within 200ms.",
            "System shall handle 10 million daily active users.",
            "Content moderation AI shall process posts within 5 minutes of upload.",
            "GDPR and COPPA compliance mandatory.",
        ],
        "stakeholders": [
            Stakeholder("Tina", "general_user", "engaging relevant content and privacy control", "no harassment", "advertiser"),
            Stakeholder("Uma", "advertiser", "targeted ad delivery and reach metrics", "brand safety", "user"),
            Stakeholder("Vince", "trust_safety_lead", "rapid harmful content removal", "false positive minimisation", None),
        ],
        "domain_entities": ["User", "Post", "Comment", "Like", "Follow", "DirectMessage", "Report", "Notification"],
        "conflicts": [
            {"req_a": "Users shall have complete control over who can see their posts.",
             "req_b": "Public interest content may be amplified by the platform beyond user privacy settings.",
             "type": "privacy_public_interest_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "event_management_platform",
        "rough_idea": "A platform for organising, promoting, and managing tickets for concerts, conferences, and community events.",
        "ground_truth_reqs": [
            "The system shall allow organisers to create event pages with schedule, venue, and speaker details.",
            "The system shall support free and paid ticket tiers with configurable capacity limits.",
            "The system shall process ticket payments and send e-tickets with QR codes via email.",
            "The system shall allow attendees to check in via QR code scan at the venue.",
            "The system shall provide organisers with real-time ticket sales and attendance dashboards.",
            "The system shall support waitlists when events are sold out and notify on cancellations.",
            "The system shall send event reminders to attendees 24 hours and 1 hour before the event.",
            "The system shall support refund processing up to a configurable cut-off date.",
        ],
        "nfr": [
            "Ticket purchase shall complete within 5 seconds to prevent drop-off.",
            "QR code check-in shall support 500 scans per minute.",
            "System shall handle traffic spikes of 10x average when popular events go on sale.",
            "PCI-DSS compliance for payment processing.",
        ],
        "stakeholders": [
            Stakeholder("Wendy", "organiser", "easy setup and full sales visibility", "low platform fees", "attendee"),
            Stakeholder("Xavier", "attendee", "smooth ticket purchase and entry experience", "reliable refunds", "organiser"),
            Stakeholder("Yolanda", "venue_manager", "accurate capacity management", "smooth entry flow", None),
        ],
        "domain_entities": ["Event", "Ticket", "Organiser", "Attendee", "Payment", "Refund", "CheckIn", "Waitlist"],
        "conflicts": [
            {"req_a": "Organisers shall set their own refund policy including non-refundable tickets.",
             "req_b": "Consumer regulations require a 14-day cooling-off period for all online ticket purchases.",
             "type": "policy_regulation_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "document_collaboration",
        "rough_idea": "A cloud platform for teams to create, edit, and collaborate on documents in real time.",
        "ground_truth_reqs": [
            "The system shall allow users to create documents, spreadsheets, and presentation files.",
            "The system shall enable simultaneous real-time co-editing with conflict-free merging.",
            "The system shall maintain a full version history with the ability to restore any past version.",
            "The system shall allow document owners to set view, comment, or edit permissions per user.",
            "The system shall support inline comments with threaded replies and resolution tracking.",
            "The system shall allow users to share documents via link with configurable access levels.",
            "The system shall support offline editing with automatic sync on reconnection.",
            "The system shall integrate with cloud storage providers for import and export.",
        ],
        "nfr": [
            "Real-time collaboration edits shall propagate within 300ms.",
            "System shall support documents up to 100MB.",
            "99.9% monthly uptime SLA.",
            "Data shall be encrypted at rest and in transit.",
        ],
        "stakeholders": [
            Stakeholder("Zara", "knowledge_worker", "fast reliable co-editing", "version control", None),
            Stakeholder("Adam", "it_admin", "enterprise security controls and SSO", "audit logging", None),
            Stakeholder("Beth", "team_manager", "visibility of document activity", "access control", None),
        ],
        "domain_entities": ["User", "Document", "Version", "Comment", "Permission", "SharedLink", "Folder", "Edit"],
        "conflicts": [
            {"req_a": "Deleted documents shall be permanently removed immediately.",
             "req_b": "Regulatory compliance requires all documents to be retained for 5 years.",
             "type": "deletion_retention_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 7 CONTINUED: GOVERNMENT & PUBLIC ──────────────────────────

    {
        "domain": "voting_system",
        "rough_idea": "A secure digital voting platform for national or local elections that ensures integrity and auditability.",
        "ground_truth_reqs": [
            "The system shall authenticate voters using government-issued digital identity before allowing a vote.",
            "The system shall ensure each eligible voter can cast exactly one ballot.",
            "The system shall anonymise votes such that no vote can be linked back to a specific voter.",
            "The system shall produce a paper audit trail for each vote cast.",
            "The system shall allow independent observers to audit the election results without compromising anonymity.",
            "The system shall publish cryptographically verifiable results within 4 hours of poll close.",
            "The system shall remain operational under DDoS attacks with 99.999% uptime during voting.",
            "The system shall support accessible voting via screen readers and alternative input devices.",
        ],
        "nfr": [
            "All vote transmissions shall use end-to-end encryption.",
            "System shall undergo independent security audit before every election.",
            "Architecture shall be open-source for public scrutiny.",
            "WCAG 2.2 AAA accessibility compliance.",
        ],
        "stakeholders": [
            Stakeholder("Chris", "eligible_voter", "simple private voting experience", "confidence in result integrity", None),
            Stakeholder("Diana", "electoral_commission", "tamper-proof audit trail", "voter anonymity", None),
            Stakeholder("Evan", "security_researcher", "public verifiability", "vulnerability disclosure", None),
        ],
        "domain_entities": ["Voter", "Ballot", "Candidate", "ElectionEvent", "AuditLog", "Result", "Observer", "DigitalIdentity"],
        "conflicts": [
            {"req_a": "Votes shall be fully anonymous with no linkage to voter identity.",
             "req_b": "Audit requirements mandate that each vote can be traced to a valid registered voter.",
             "type": "anonymity_auditability_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "emergency_response_system",
        "rough_idea": "A system for coordinating emergency services response to incidents across police, fire, and ambulance.",
        "ground_truth_reqs": [
            "The system shall receive emergency calls and capture incident details including location and type.",
            "The system shall automatically determine the closest available appropriate unit using GPS data.",
            "The system shall dispatch units with turn-by-turn routing to the incident location.",
            "The system shall allow dispatchers to monitor all unit locations and statuses on a live map.",
            "The system shall escalate incidents automatically when response time thresholds are breached.",
            "The system shall allow multi-agency coordination for major incidents with shared incident logs.",
            "The system shall record all dispatch communications for post-incident review.",
            "The system shall provide offline capability for field units when network connectivity is unavailable.",
        ],
        "nfr": [
            "Dispatch decision shall be made within 60 seconds of call receipt.",
            "System shall achieve 99.999% uptime.",
            "Field unit app shall function on 2G networks as minimum.",
            "All data shall be stored in sovereign national infrastructure.",
        ],
        "stakeholders": [
            Stakeholder("Frank", "dispatcher", "clear incident overview and fast dispatch tools", "unit safety", None),
            Stakeholder("Gina", "paramedic", "accurate routing and patient information en route", "personal safety", None),
            Stakeholder("Henry", "emergency_director", "cross-agency coordination and KPI reporting", "budget compliance", None),
        ],
        "domain_entities": ["Incident", "Unit", "Dispatcher", "Location", "ResourceAllocation", "Communication", "AgencyLog", "Route"],
        "conflicts": [
            {"req_a": "System shall automatically dispatch the closest unit to minimise response time.",
             "req_b": "Some incidents require specialist units that may be further away, overriding proximity logic.",
             "type": "automation_specialisation_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "public_transport_app",
        "rough_idea": "A mobile app for passengers to plan journeys, buy tickets, and track buses and trains in real time.",
        "ground_truth_reqs": [
            "The system shall provide multi-modal journey planning across bus, rail, metro, and walking.",
            "The system shall show real-time vehicle locations and predicted arrival times at stops.",
            "The system shall allow passengers to purchase single, day, and season tickets within the app.",
            "The system shall support contactless QR code ticket validation at gates and on vehicles.",
            "The system shall send service disruption alerts and suggest alternative routes.",
            "The system shall allow passengers to set favourite journeys and receive proactive updates.",
            "The system shall provide accessibility journey options for wheelchair users.",
            "The system shall integrate with national rail and regional transport authority data feeds.",
        ],
        "nfr": [
            "Real-time arrival predictions shall update every 30 seconds.",
            "App shall load journey results within 3 seconds.",
            "System shall comply with UK open data standards for transport.",
            "WCAG 2.1 AA accessibility compliance.",
        ],
        "stakeholders": [
            Stakeholder("Iris", "commuter", "accurate real-time information and easy ticketing", "journey time saving", None),
            Stakeholder("Jake", "transport_authority", "ridership growth and revenue", "data accuracy", None),
            Stakeholder("Karen", "accessibility_advocate", "full wheelchair and disability journey planning", "step-free routes", None),
        ],
        "domain_entities": ["Journey", "Ticket", "Vehicle", "Stop", "Route", "Disruption", "Passenger", "ServiceFeed"],
        "conflicts": [
            {"req_a": "App shall display the fastest route regardless of accessibility.",
             "req_b": "Accessibility settings shall always override speed optimisation for eligible users.",
             "type": "optimisation_accessibility_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 8 CONTINUED: ENTERTAINMENT & MEDIA ────────────────────────

    {
        "domain": "music_streaming",
        "rough_idea": "A music streaming service where users discover, play, and curate songs and playlists.",
        "ground_truth_reqs": [
            "The system shall provide a catalogue of over 100 million tracks searchable by artist, album, and genre.",
            "The system shall stream audio at 128kbps standard and 320kbps premium quality.",
            "The system shall allow users to create, edit, and share playlists.",
            "The system shall generate personalised daily mix playlists based on listening history.",
            "The system shall allow offline download of up to 10,000 tracks for premium subscribers.",
            "The system shall display real-time lyrics synchronised to playback.",
            "The system shall support cross-device playback handoff with seamless continuation.",
            "The system shall accurately report stream counts to rights holders for royalty calculation.",
        ],
        "nfr": [
            "Audio playback shall start within 1 second.",
            "System shall handle 50 million concurrent streams.",
            "DRM protection shall be applied to all downloaded tracks.",
            "Royalty data accuracy shall meet IFPI audit standards.",
        ],
        "stakeholders": [
            Stakeholder("Liam", "listener", "vast library and personalised discovery", "offline access", None),
            Stakeholder("Mia", "artist", "fair royalty payments and fan analytics", "creative control", "platform"),
            Stakeholder("Noah", "label_representative", "DRM enforcement and accurate reporting", "new release promotion", None),
        ],
        "domain_entities": ["Track", "Artist", "Album", "Playlist", "User", "StreamEvent", "Download", "Royalty"],
        "conflicts": [
            {"req_a": "Personalised recommendations shall maximise listening time by suggesting familiar artists.",
             "req_b": "Artist discovery features shall expose users to new artists, reducing familiar content.",
             "type": "engagement_discovery_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "gaming_platform",
        "rough_idea": "A PC and console gaming platform for purchasing, downloading, and playing games with social features.",
        "ground_truth_reqs": [
            "The system shall allow users to purchase and download games directly to their device.",
            "The system shall provide a friends list with online status and in-game activity.",
            "The system shall support in-game overlay for chat, notifications, and invites.",
            "The system shall manage automatic game updates in the background.",
            "The system shall provide a community hub per game with reviews, screenshots, and forums.",
            "The system shall support achievements and a public profile showcasing completed games.",
            "The system shall enforce age verification and parental controls for age-rated content.",
            "The system shall provide a refund policy allowing claims within 2 hours of playtime within 14 days.",
        ],
        "nfr": [
            "Download speeds shall saturate available bandwidth without throttling.",
            "Platform client shall use under 2% CPU at idle.",
            "System shall handle 30 million concurrent users.",
            "Fraud detection on purchase shall add under 500ms latency.",
        ],
        "stakeholders": [
            Stakeholder("Oliver", "gamer", "large library and smooth download experience", "competitive community features", None),
            Stakeholder("Priya", "game_developer", "revenue share and review visibility", "anti-piracy protection", "gamer"),
            Stakeholder("Quinn", "platform_operator", "marketplace revenue and user retention", "developer relations", None),
        ],
        "domain_entities": ["User", "Game", "Purchase", "Download", "Friend", "Achievement", "Review", "Refund"],
        "conflicts": [
            {"req_a": "Refunds shall be granted automatically if playtime is under 2 hours.",
             "req_b": "Developers shall have the right to dispute refund requests for games with clear playtime abuse.",
             "type": "consumer_rights_developer_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "podcast_platform",
        "rough_idea": "A platform where creators publish podcasts and listeners discover, subscribe, and play episodes.",
        "ground_truth_reqs": [
            "The system shall allow creators to upload audio files and publish episodes with titles and show notes.",
            "The system shall generate an RSS feed for each podcast for cross-platform distribution.",
            "The system shall allow listeners to search, subscribe, and receive new episode notifications.",
            "The system shall track listening progress and resume playback from the last position.",
            "The system shall support variable playback speed from 0.5x to 3x.",
            "The system shall allow creators to access episode analytics including plays, completion rate, and geography.",
            "The system shall support chapter markers for navigation within long episodes.",
            "The system shall allow premium subscription gating for exclusive episodes.",
        ],
        "nfr": [
            "Audio shall start streaming within 2 seconds.",
            "System shall support uploads up to 500MB per episode.",
            "RSS feeds shall update within 5 minutes of episode publication.",
            "GDPR compliance for listener analytics data.",
        ],
        "stakeholders": [
            Stakeholder("Ruby", "podcast_creator", "audience growth and analytics", "monetisation tools", None),
            Stakeholder("Sam", "listener", "easy discovery and offline listening", "no intrusive ads", None),
            Stakeholder("Tara", "platform_manager", "creator and listener retention", "premium revenue", None),
        ],
        "domain_entities": ["Podcast", "Episode", "Creator", "Listener", "Subscription", "RSSFeed", "Analytics", "Chapter"],
        "conflicts": [
            {"req_a": "Creator analytics shall include detailed per-listener engagement data.",
             "req_b": "GDPR prohibits sharing individually identifiable listening behaviour with third parties.",
             "type": "analytics_privacy_conflict"},
        ],
        "difficulty": "easy",
    },

    # ── SECTOR 9: HR & WORKFORCE ─────────────────────────────────────────

    {
        "domain": "recruitment_platform",
        "rough_idea": "An applicant tracking system for companies to post jobs, manage applications, and conduct hiring.",
        "ground_truth_reqs": [
            "The system shall allow recruiters to create and publish job postings to multiple job boards.",
            "The system shall parse CVs automatically to extract candidate skills and experience.",
            "The system shall screen applications against job criteria and provide a match score.",
            "The system shall manage interview scheduling with calendar integration for all participants.",
            "The system shall allow interviewers to submit structured scorecards after each interview.",
            "The system shall maintain a talent pool for future roles with candidate consent.",
            "The system shall send automated status notifications to candidates at each stage.",
            "The system shall generate equal opportunity hiring reports to detect bias.",
        ],
        "nfr": [
            "CV parsing shall complete within 10 seconds.",
            "System shall comply with GDPR and UK Equality Act for candidate data.",
            "System shall integrate with LinkedIn, Indeed, and major ATS platforms.",
            "Candidate data shall be deleted after 2 years unless consent is renewed.",
        ],
        "stakeholders": [
            Stakeholder("Uma", "recruiter", "efficient pipeline management and fast time-to-hire", "quality shortlists", None),
            Stakeholder("Victor", "hiring_manager", "structured interview data and quick decisions", "diverse shortlists", "recruiter"),
            Stakeholder("Wendy", "candidate", "transparent process and timely feedback", "privacy of CV data", None),
        ],
        "domain_entities": ["Job", "Candidate", "Application", "Interview", "Scorecard", "TalentPool", "Offer", "Report"],
        "conflicts": [
            {"req_a": "AI matching shall auto-reject applications below a threshold score to save recruiter time.",
             "req_b": "All rejections shall be reviewed by a human to prevent algorithmic bias.",
             "type": "efficiency_fairness_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "hr_management_system",
        "rough_idea": "A comprehensive HR platform covering employee records, leave management, payroll, and performance.",
        "ground_truth_reqs": [
            "The system shall maintain a digital employee record including personal details, role, and contract.",
            "The system shall manage the full employee lifecycle from onboarding to offboarding.",
            "The system shall process monthly payroll with tax, NI, and benefit deductions.",
            "The system shall manage leave requests, approvals, and entitlement balances.",
            "The system shall generate payslips and P60s electronically.",
            "The system shall track training and certifications per employee with expiry alerts.",
            "The system shall support organisational chart visualisation and reporting lines.",
            "The system shall generate headcount, attrition, and diversity analytics reports.",
        ],
        "nfr": [
            "Payroll calculations shall be 100% accurate with zero tolerance for errors.",
            "System shall comply with HMRC RTI reporting requirements.",
            "Employee personal data shall comply with GDPR.",
            "System shall support multi-currency payroll for international employees.",
        ],
        "stakeholders": [
            Stakeholder("Xena", "hr_business_partner", "accurate employee data and easy reporting", "compliance", None),
            Stakeholder("Yuri", "employee", "transparent payslips and easy leave booking", "privacy", None),
            Stakeholder("Zoe", "payroll_manager", "accurate automated payroll and HMRC compliance", "audit trail", None),
        ],
        "domain_entities": ["Employee", "Contract", "Leave", "Payroll", "Payslip", "Training", "Department", "OrgChart"],
        "conflicts": [
            {"req_a": "Employee records shall be accessible to all HR team members.",
             "req_b": "Sensitive data such as salary and disciplinary records shall be restricted to senior HR only.",
             "type": "access_sensitivity_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "employee_performance_management",
        "rough_idea": "A platform for setting goals, conducting performance reviews, and managing feedback between managers and employees.",
        "ground_truth_reqs": [
            "The system shall allow employees to set OKRs and KPIs aligned to company objectives.",
            "The system shall support continuous feedback with praise and development notes.",
            "The system shall facilitate mid-year and annual review cycles with structured templates.",
            "The system shall enable 360-degree feedback from peers, reports, and managers.",
            "The system shall provide managers with a team performance heatmap.",
            "The system shall support calibration sessions where managers align ratings across departments.",
            "The system shall generate development plan recommendations based on review outcomes.",
            "The system shall produce talent matrix reports for succession planning.",
        ],
        "nfr": [
            "Review data shall be encrypted at rest.",
            "System shall support single sign-on via SAML 2.0.",
            "GDPR compliance for performance data.",
            "System shall handle reviews for organisations up to 50,000 employees.",
        ],
        "stakeholders": [
            Stakeholder("Aaron", "employee", "transparent fair assessment and development support", "privacy of feedback", "manager"),
            Stakeholder("Beth", "line_manager", "structured review tools and team visibility", "calibration support", "employee"),
            Stakeholder("Carl", "hr_director", "consistent ratings and succession pipeline", "legal compliance", None),
        ],
        "domain_entities": ["Employee", "Goal", "Review", "Feedback", "Rating", "CalibrationSession", "DevelopmentPlan", "Report"],
        "conflicts": [
            {"req_a": "360-degree feedback shall be anonymous to encourage honesty.",
             "req_b": "Managers shall be able to identify feedback sources to address unfair comments.",
             "type": "anonymity_accountability_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "workforce_scheduling",
        "rough_idea": "A shift scheduling and workforce management system for retail, hospitality, and care sector employers.",
        "ground_truth_reqs": [
            "The system shall generate optimised shift schedules based on demand forecasts and staff availability.",
            "The system shall allow employees to submit availability and time-off requests.",
            "The system shall notify employees of their schedules at least 72 hours in advance.",
            "The system shall allow employees to swap shifts with manager approval.",
            "The system shall enforce working time regulations including maximum hours and rest period rules.",
            "The system shall integrate with payroll to calculate hours and overtime automatically.",
            "The system shall track actual vs scheduled hours and flag discrepancies.",
            "The system shall send real-time alerts when understaffed shifts are detected.",
        ],
        "nfr": [
            "Schedule generation shall complete within 60 seconds for 500 employees.",
            "Mobile app shall support iOS and Android for shift viewing and swaps.",
            "System shall comply with Working Time Regulations 1998.",
            "Integration with major EPOS and payroll systems via API.",
        ],
        "stakeholders": [
            Stakeholder("Dana", "shift_worker", "fair predictable schedules and easy swaps", "work-life balance", "manager"),
            Stakeholder("Ed", "store_manager", "fully staffed shifts and minimal manual intervention", "labour cost control", "worker"),
            Stakeholder("Fay", "hr_compliance", "working time regulation adherence", "audit trail", None),
        ],
        "domain_entities": ["Employee", "Shift", "Schedule", "Availability", "SwapRequest", "Department", "Payroll", "Alert"],
        "conflicts": [
            {"req_a": "System shall auto-approve shift swaps to minimise manager admin.",
             "req_b": "Managers shall approve all swaps to ensure skill requirements per shift are met.",
             "type": "automation_control_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 10: REAL ESTATE & PROPERTY ────────────────────────────────

    {
        "domain": "property_listing_platform",
        "rough_idea": "An online property marketplace where estate agents and private sellers list homes for sale or rent.",
        "ground_truth_reqs": [
            "The system shall allow agents and private sellers to create property listings with photos and floor plans.",
            "The system shall allow buyers and renters to search by location, price, bedrooms, and property type.",
            "The system shall display properties on an interactive map with polygon-based search.",
            "The system shall allow registered users to save properties and receive price change alerts.",
            "The system shall allow interested parties to request viewings directly through the platform.",
            "The system shall provide a mortgage calculator and stamp duty estimator.",
            "The system shall display Energy Performance Certificate ratings on all listings.",
            "The system shall allow agents to manage their listing portfolio and track enquiry volumes.",
        ],
        "nfr": [
            "Search results shall return within 1 second for filters on 5 million listings.",
            "Photo galleries shall support lazy loading for listings with 50+ images.",
            "System shall comply with GDPR for user search and enquiry data.",
            "Listing data feeds from portals shall sync every 15 minutes.",
        ],
        "stakeholders": [
            Stakeholder("Gill", "home_buyer", "accurate listings and easy viewing requests", "neighbourhood insights", None),
            Stakeholder("Hugh", "estate_agent", "lead generation and enquiry management", "low portal fees", "buyer"),
            Stakeholder("Ida", "platform_manager", "listing volume and user engagement", "data quality", None),
        ],
        "domain_entities": ["Property", "Listing", "Agent", "Buyer", "Enquiry", "ViewingRequest", "SavedSearch", "EPC"],
        "conflicts": [
            {"req_a": "Private sellers shall list without agent involvement to reduce costs.",
             "req_b": "The platform requires agent validation to ensure listing accuracy and legal compliance.",
             "type": "access_quality_conflict"},
        ],
        "difficulty": "easy",
    },

    {
        "domain": "property_management_system",
        "rough_idea": "A system for landlords and letting agents to manage tenancies, maintenance, and rent collection.",
        "ground_truth_reqs": [
            "The system shall manage tenancy agreements with digital signing and secure storage.",
            "The system shall track rent collection with automated payment reminders for overdue rent.",
            "The system shall allow tenants to log maintenance requests with photos and priority.",
            "The system shall assign maintenance jobs to contractors and track completion.",
            "The system shall manage deposit protection scheme registration and return workflows.",
            "The system shall generate gas safety and EPC certificate expiry alerts.",
            "The system shall provide landlord financial reports including income, expenses, and tax summaries.",
            "The system shall support portfolio management for landlords with multiple properties.",
        ],
        "nfr": [
            "System shall comply with UK tenant deposit protection regulations.",
            "All signed documents shall use legally binding e-signatures.",
            "System shall support GDPR for tenant personal data.",
            "Financial reports shall export to CSV and PDF formats.",
        ],
        "stakeholders": [
            Stakeholder("Jake", "landlord", "automated rent collection and compliance alerts", "low vacancy periods", "tenant"),
            Stakeholder("Kate", "tenant", "fast maintenance resolution and transparent deposit handling", "privacy", "landlord"),
            Stakeholder("Leo", "letting_agent", "portfolio oversight and inspection scheduling", "commission management", None),
        ],
        "domain_entities": ["Property", "Tenancy", "Tenant", "Landlord", "MaintenanceRequest", "Deposit", "Certificate", "Payment"],
        "conflicts": [
            {"req_a": "Tenants shall have 24/7 ability to log urgent maintenance as emergency.",
             "req_b": "Landlords shall classify what constitutes an emergency to prevent abuse of out-of-hours call-outs.",
             "type": "urgency_classification_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "construction_project_management",
        "rough_idea": "A platform to plan, track, and manage construction projects across multiple sites and contractors.",
        "ground_truth_reqs": [
            "The system shall manage project plans with tasks, milestones, and dependencies on a Gantt chart.",
            "The system shall track labour, materials, and equipment costs against the project budget.",
            "The system shall manage RFI and submittal workflows between stakeholders.",
            "The system shall allow site managers to log daily progress and issues via mobile app.",
            "The system shall manage drawing version control with clash detection alerts.",
            "The system shall track safety incidents and near-misses with investigation workflows.",
            "The system shall generate progress reports and cost variance reports for project owners.",
            "The system shall manage subcontractor and supplier contracts and payment schedules.",
        ],
        "nfr": [
            "Drawing files shall support DWG, PDF, and IFC BIM formats up to 500MB.",
            "Mobile app shall work offline on site with sync when connected.",
            "System shall comply with CDM Construction Design and Management regulations.",
            "All document versions shall be immutably logged with timestamps.",
        ],
        "stakeholders": [
            Stakeholder("Mike", "project_manager", "schedule and budget control", "risk management", None),
            Stakeholder("Nora", "site_manager", "real-time issue logging and drawing access", "safety compliance", None),
            Stakeholder("Owen", "client", "cost transparency and milestone visibility", "on-time delivery", None),
        ],
        "domain_entities": ["Project", "Task", "Drawing", "RFI", "Subcontractor", "CostItem", "SafetyIncident", "Milestone"],
        "conflicts": [
            {"req_a": "Design changes shall be approved by the client before implementation.",
             "req_b": "Site-critical design changes must be actioned immediately to avoid work stoppage.",
             "type": "approval_urgency_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 11: AGRICULTURE & ENVIRONMENT ─────────────────────────────

    {
        "domain": "precision_farming_platform",
        "rough_idea": "An IoT and analytics platform to help farmers optimise crop yields and reduce input costs.",
        "ground_truth_reqs": [
            "The system shall collect soil moisture, temperature, and nutrient data from field sensors.",
            "The system shall integrate with weather forecast APIs to adjust irrigation recommendations.",
            "The system shall generate variable-rate application maps for fertiliser and pesticides.",
            "The system shall track crop growth stages using satellite and drone imagery analysis.",
            "The system shall provide yield prediction models per field based on historical and current data.",
            "The system shall allow farmers to log field operations including planting, spraying, and harvesting.",
            "The system shall generate compliance reports for farm assurance schemes.",
            "The system shall send alerts when soil conditions or weather forecast indicate crop risk.",
        ],
        "nfr": [
            "Sensor data ingestion shall support 1 million readings per hour.",
            "Satellite imagery processing shall complete within 2 hours of data receipt.",
            "Mobile app shall function offline in areas with no connectivity.",
            "System shall comply with GDPR for farmer business data.",
        ],
        "stakeholders": [
            Stakeholder("Pat", "farmer", "yield improvement and input cost reduction", "easy data entry", None),
            Stakeholder("Quinn", "agronomist", "accurate field data for recommendations", "remote advisory capability", None),
            Stakeholder("Ruth", "agricultural_retailer", "demand forecasting for inputs", "farmer relationship", None),
        ],
        "domain_entities": ["Field", "Sensor", "CropType", "SoilReading", "IrrigationPlan", "YieldPrediction", "FieldOperation", "Alert"],
        "conflicts": [
            {"req_a": "Farmer field data shall be shared with agronomist partners for advisory services.",
             "req_b": "Farmers retain ownership of their data and must consent before any sharing.",
             "type": "data_sharing_ownership_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "environmental_monitoring",
        "rough_idea": "A platform to monitor air quality, water quality, and noise levels across a city and alert authorities to breaches.",
        "ground_truth_reqs": [
            "The system shall ingest sensor readings for PM2.5, NO2, CO2, and noise from a network of stations.",
            "The system shall display real-time air quality index on a public-facing map.",
            "The system shall trigger automated alerts to the environmental authority when WHO thresholds are exceeded.",
            "The system shall correlate pollution events with weather data and traffic patterns.",
            "The system shall generate daily, weekly, and annual air quality reports per monitoring zone.",
            "The system shall allow the public to report pollution incidents via a citizen reporting feature.",
            "The system shall support integration with national air quality networks for data contribution.",
            "The system shall predict air quality levels for the next 48 hours using a forecasting model.",
        ],
        "nfr": [
            "Sensor data shall be ingested and visualised within 60 seconds.",
            "System shall handle a network of 500 monitoring stations.",
            "Public API shall provide open data access per Open Government Licence.",
            "Forecasting model accuracy shall be within 20% of measured values.",
        ],
        "stakeholders": [
            Stakeholder("Sam", "environmental_officer", "threshold breach alerts and compliance reporting", "data accuracy", None),
            Stakeholder("Tess", "public_citizen", "accessible real-time air quality information", "health advice", None),
            Stakeholder("Uri", "city_planner", "long-term trend analysis for policy decisions", "cross-sector data", None),
        ],
        "domain_entities": ["Sensor", "Station", "Reading", "Alert", "PollutantType", "Zone", "PublicReport", "Forecast"],
        "conflicts": [
            {"req_a": "All raw sensor data shall be published openly for research.",
             "req_b": "Industrial operators near monitoring stations require advance notice before data publication.",
             "type": "openness_commercial_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "supply_chain_traceability",
        "rough_idea": "A blockchain-based platform to track products from farm to shelf and verify ethical sourcing.",
        "ground_truth_reqs": [
            "The system shall record each supply chain event as an immutable entry covering harvest, processing, shipping, and retail.",
            "The system shall allow consumers to scan a product QR code to view its full provenance journey.",
            "The system shall verify supplier certifications against a trusted registry.",
            "The system shall detect and flag anomalies in the supply chain such as missing steps or time gaps.",
            "The system shall support multi-party access for producers, processors, distributors, and retailers.",
            "The system shall generate audit-ready traceability reports for food safety inspections.",
            "The system shall integrate with IoT sensors for automated recording of storage conditions.",
            "The system shall support product recall workflows by identifying affected batch distribution.",
        ],
        "nfr": [
            "QR code provenance lookup shall respond within 2 seconds.",
            "Blockchain transaction finality shall occur within 5 seconds.",
            "System shall handle 10 million product batches.",
            "GDPR compliance for supplier personal data.",
        ],
        "stakeholders": [
            Stakeholder("Val", "retailer", "supplier compliance and recall capability", "consumer trust", "supplier"),
            Stakeholder("Will", "supplier", "low compliance overhead and data privacy", "fair trade certification", "retailer"),
            Stakeholder("Xia", "food_safety_regulator", "complete traceable audit trail", "rapid recall response", None),
        ],
        "domain_entities": ["Product", "Batch", "SupplyEvent", "Supplier", "Certification", "Blockchain", "QRCode", "RecallAlert"],
        "conflicts": [
            {"req_a": "Full supply chain data shall be visible to consumers for transparency.",
             "req_b": "Suppliers require commercial data such as quantities and pricing to be redacted from public view.",
             "type": "transparency_confidentiality_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 12: LEGAL & COMPLIANCE ────────────────────────────────────

    {
        "domain": "legal_case_management",
        "rough_idea": "A practice management system for law firms to manage cases, client communications, and billing.",
        "ground_truth_reqs": [
            "The system shall allow fee earners to create matters linked to clients with conflict-of-interest checks.",
            "The system shall manage court dates, deadlines, and limitation periods with calendar alerts.",
            "The system shall store all case correspondence and documents with version control.",
            "The system shall track time recorded against each matter for billing purposes.",
            "The system shall generate invoices and fee notes from recorded time and disbursements.",
            "The system shall allow secure client portal access to share documents and updates.",
            "The system shall support electronic bundling for court submissions.",
            "The system shall produce regulatory compliance reports for SRA and AML obligations.",
        ],
        "nfr": [
            "Conflict-of-interest check shall run within 5 seconds against the full client database.",
            "System shall comply with SRA Standards and Regulations.",
            "All documents shall be backed up with 99.99% durability.",
            "Client portal communications shall be end-to-end encrypted.",
        ],
        "stakeholders": [
            Stakeholder("Yvonne", "solicitor", "efficient matter management and billing capture", "client satisfaction", "client"),
            Stakeholder("Zack", "client", "transparent case progress and secure document exchange", "cost certainty", "solicitor"),
            Stakeholder("Ann", "practice_manager", "financial performance and regulatory compliance", "resource utilisation", None),
        ],
        "domain_entities": ["Matter", "Client", "Document", "TimeEntry", "Invoice", "CourtDate", "Bundle", "ComplianceReport"],
        "conflicts": [
            {"req_a": "Clients shall have real-time access to all case documents via the portal.",
             "req_b": "Solicitors shall retain the right to withhold documents subject to legal professional privilege.",
             "type": "access_privilege_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "contract_management",
        "rough_idea": "A system for businesses to draft, negotiate, execute, and manage the lifecycle of commercial contracts.",
        "ground_truth_reqs": [
            "The system shall provide a template library for standard contract types with clause libraries.",
            "The system shall support collaborative redlining with tracked changes between parties.",
            "The system shall manage approval workflows with configurable signatory levels.",
            "The system shall execute contracts via legally binding electronic signatures.",
            "The system shall track key dates including expiry, renewal, and obligation milestones.",
            "The system shall send automated alerts for upcoming renewals and obligations.",
            "The system shall provide a contract repository with full-text search across all clauses.",
            "The system shall generate contract analytics including cycle time and clause deviation reports.",
        ],
        "nfr": [
            "Full-text search across 1 million contracts shall return results within 3 seconds.",
            "eSignature shall comply with EU eIDAS and US ESIGN Act.",
            "System shall integrate with Salesforce and SAP for CRM and ERP data.",
            "All contracts shall be retained for their legal retention period.",
        ],
        "stakeholders": [
            Stakeholder("Brad", "contract_manager", "fast cycle time and obligation visibility", "clause standardisation", "legal"),
            Stakeholder("Claire", "legal_counsel", "risk review and clause control", "playbook compliance", "business"),
            Stakeholder("Dave", "procurement_manager", "supplier contract compliance", "cost visibility", None),
        ],
        "domain_entities": ["Contract", "Template", "Clause", "Party", "Approval", "Signature", "Obligation", "Amendment"],
        "conflicts": [
            {"req_a": "Business teams shall finalise contracts directly with minimal legal review to speed deals.",
             "req_b": "Legal team requires review of all contracts above £100,000 before execution.",
             "type": "speed_governance_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 13: TRAVEL & HOSPITALITY ──────────────────────────────────

    {
        "domain": "travel_booking_platform",
        "rough_idea": "An online travel agency where users can search and book flights, hotels, and holiday packages.",
        "ground_truth_reqs": [
            "The system shall allow users to search for flights by origin, destination, dates, and passenger count.",
            "The system shall display fare comparison across airlines with filters for stops and duration.",
            "The system shall support hotel search with map view, photos, and reviews.",
            "The system shall allow bundling of flights and hotels into a package with combined pricing.",
            "The system shall process payments and send booking confirmations with e-tickets.",
            "The system shall manage booking amendments and cancellations per airline and hotel policies.",
            "The system shall send check-in reminders and travel documentation alerts.",
            "The system shall provide a post-trip review feature for flights, hotels, and packages.",
        ],
        "nfr": [
            "Flight search shall return results within 3 seconds from live GDS data.",
            "System shall comply with EU Package Travel Directive.",
            "PCI-DSS compliance for payment processing.",
            "System shall handle 100,000 concurrent searches.",
        ],
        "stakeholders": [
            Stakeholder("Ella", "traveller", "best price and seamless booking", "flexible cancellation", "supplier"),
            Stakeholder("Fred", "airline_partner", "accurate inventory distribution and booking fees", "ancillary upsell", "traveller"),
            Stakeholder("Gina", "operations_manager", "supplier contract compliance and refund management", "customer satisfaction", None),
        ],
        "domain_entities": ["Flight", "Hotel", "Package", "Booking", "Passenger", "Payment", "Cancellation", "Review"],
        "conflicts": [
            {"req_a": "Platform shall advertise lowest fares prominently to attract price-sensitive customers.",
             "req_b": "Airlines require surcharges and fees to be displayed before payment not after selection.",
             "type": "marketing_transparency_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "hotel_management_system",
        "rough_idea": "A property management system for hotels to manage reservations, housekeeping, and guest services.",
        "ground_truth_reqs": [
            "The system shall manage room inventory and accept reservations from direct and OTA channels.",
            "The system shall perform automatic rate management based on occupancy and demand.",
            "The system shall manage check-in and check-out processes including key card assignment.",
            "The system shall coordinate housekeeping schedules based on checkout and arrival patterns.",
            "The system shall manage guest folios and post charges from restaurant and room service.",
            "The system shall support group bookings with block allocation and rooming list management.",
            "The system shall send pre-arrival and post-departure communication sequences.",
            "The system shall generate front office, revenue, and housekeeping operational reports.",
        ],
        "nfr": [
            "Reservation updates shall synchronise across all channels within 60 seconds to prevent overbooking.",
            "System shall comply with PCI-DSS for card payment storage.",
            "System shall integrate with OTAs via HTNG and OTA XML standards.",
            "System uptime shall be 99.99% to prevent front-desk disruption.",
        ],
        "stakeholders": [
            Stakeholder("Hugh", "hotel_manager", "full occupancy and revenue optimisation", "guest satisfaction", None),
            Stakeholder("Iris", "front_desk_agent", "fast check-in and real-time room status", "guest conflict resolution", None),
            Stakeholder("Jack", "revenue_manager", "dynamic pricing and OTA parity", "direct booking ratio", None),
        ],
        "domain_entities": ["Room", "Reservation", "Guest", "Folio", "HousekeepingTask", "Rate", "Channel", "GroupBlock"],
        "conflicts": [
            {"req_a": "OTA channels shall display real-time availability to maximise bookings.",
             "req_b": "Hotel shall maintain rate parity across all channels to avoid OTA contract violations.",
             "type": "distribution_parity_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 14: FOOD & LIFESTYLE ───────────────────────────────────────

    {
        "domain": "restaurant_pos_system",
        "rough_idea": "A point-of-sale system for restaurants to take orders, manage tables, and process payments.",
        "ground_truth_reqs": [
            "The system shall allow servers to take table orders and send them directly to the kitchen display.",
            "The system shall manage table layouts with status indicators for occupied, reserved, and available.",
            "The system shall support split bill functionality by items, evenly, or by percentage.",
            "The system shall process payments by card, cash, and digital wallet.",
            "The system shall manage menu items with modifiers, allergen information, and availability.",
            "The system shall generate end-of-shift sales reports and reconciliation summaries.",
            "The system shall integrate with kitchen display systems and receipt printers.",
            "The system shall support online ordering integration with a branded ordering page.",
        ],
        "nfr": [
            "Order transmission to kitchen shall occur within 2 seconds.",
            "System shall operate offline if internet is lost with background sync when restored.",
            "PCI-DSS compliance for card payments.",
            "Allergen information shall be prominently displayed per EU Food Information Regulation.",
        ],
        "stakeholders": [
            Stakeholder("Kim", "server", "fast order entry and table management", "minimal errors", None),
            Stakeholder("Leo", "restaurant_owner", "sales visibility and stock management", "low downtime", None),
            Stakeholder("Mel", "head_chef", "accurate order flow and kitchen timing", "allergy accuracy", None),
        ],
        "domain_entities": ["Table", "Order", "MenuItem", "Payment", "KitchenTicket", "Server", "Modifier", "SalesReport"],
        "conflicts": [
            {"req_a": "Servers shall be able to modify orders after kitchen confirmation to handle customer changes.",
             "req_b": "Once kitchen confirms an order, modifications increase waste and must be charged.",
             "type": "flexibility_waste_conflict"},
        ],
        "difficulty": "easy",
    },

    {
        "domain": "fitness_tracking_app",
        "rough_idea": "A mobile app for logging workouts, tracking nutrition, and monitoring health metrics from wearables.",
        "ground_truth_reqs": [
            "The system shall allow users to log workouts by type, duration, and intensity.",
            "The system shall sync with wearable devices to import steps, heart rate, and sleep data.",
            "The system shall provide a food diary with calorie and macro tracking via a barcode scanner.",
            "The system shall generate weekly fitness and nutrition summary reports.",
            "The system shall allow users to set fitness goals and track progress against them.",
            "The system shall provide guided workout programmes with video demonstrations.",
            "The system shall allow users to connect with friends and share achievements.",
            "The system shall detect workout inactivity streaks and send motivational prompts.",
        ],
        "nfr": [
            "Wearable sync shall complete within 30 seconds.",
            "Food barcode database shall contain over 1 million products.",
            "App shall comply with HIPAA for users who share data with healthcare providers.",
            "Battery impact shall add less than 5% drain per day for background health monitoring.",
        ],
        "stakeholders": [
            Stakeholder("Nick", "fitness_user", "accurate tracking and motivation", "health data privacy", None),
            Stakeholder("Olivia", "personal_trainer", "client progress visibility and programme delivery", "remote coaching tools", None),
            Stakeholder("Pete", "product_manager", "daily active use and premium subscription conversion", "user retention", None),
        ],
        "domain_entities": ["User", "Workout", "Exercise", "FoodEntry", "HealthMetric", "Goal", "Programme", "WearableDevice"],
        "conflicts": [
            {"req_a": "Health data shall be shared with partner healthcare providers for holistic care.",
             "req_b": "Users shall have full control over health data sharing with no default opt-in.",
             "type": "healthcare_integration_privacy_conflict"},
        ],
        "difficulty": "easy",
    },

    {
        "domain": "ride_sharing_platform",
        "rough_idea": "A platform connecting passengers with drivers for on-demand and pre-booked rides in cities.",
        "ground_truth_reqs": [
            "The system shall allow passengers to request rides by entering a destination on a map.",
            "The system shall match passengers with the nearest available driver within 2 minutes.",
            "The system shall display estimated fare before the passenger confirms the booking.",
            "The system shall allow passengers to track their driver in real time on a map.",
            "The system shall process payments automatically via stored card on trip completion.",
            "The system shall allow passengers to rate drivers and drivers to rate passengers.",
            "The system shall allow drivers to accept or decline trip requests within 30 seconds.",
            "The system shall support scheduled bookings up to 7 days in advance.",
        ],
        "nfr": [
            "Driver matching shall complete within 10 seconds.",
            "Location tracking shall update every 5 seconds.",
            "System shall comply with local taxi and private hire regulations.",
            "Surge pricing algorithm shall comply with consumer protection disclosure requirements.",
        ],
        "stakeholders": [
            Stakeholder("Quin", "passenger", "fast reliable rides at fair prices", "safety and driver ratings", "driver"),
            Stakeholder("Ray", "driver", "steady income and fair ride allocation", "passenger safety", "passenger"),
            Stakeholder("Sue", "operations_manager", "platform safety compliance and growth", "driver retention", None),
        ],
        "domain_entities": ["Passenger", "Driver", "Ride", "Route", "Payment", "Rating", "MatchingAlgorithm", "Location"],
        "conflicts": [
            {"req_a": "Surge pricing shall activate automatically during peak demand to attract more drivers.",
             "req_b": "Consumer regulations require advance transparent notice before surge pricing is applied.",
             "type": "dynamic_pricing_transparency_conflict"},
        ],
        "difficulty": "medium",
    },
]


# ─────────────────────────────────────────────
#  ScenarioGenerator
# ─────────────────────────────────────────────

class ScenarioGenerator:
    """
    Generates synthetic RE scenarios for training REMARL agents.

    Usage:
        gen = ScenarioGenerator("data/scenarios/")
        scenario = gen.sample()              # random scenario
        scenario = gen.sample(domain="healthcare")  # domain-specific
        scenario = gen.sample(difficulty="hard")    # difficulty-specific
        batch    = gen.sample_batch(n=32)    # batch for training

    The scenario_dir is used to cache generated scenarios as JSON files.
    On first run, all templates are expanded and cached.
    On subsequent runs, cached scenarios are loaded for reproducibility.
    """

    def __init__(
        self,
        scenario_dir: str = "data/scenarios/",
        hide_fraction: float = 0.25,   # fraction of reqs to hide per episode
        seed: int = 42,
    ):
        self.scenario_dir = pathlib.Path(scenario_dir)
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        self.hide_fraction = hide_fraction
        self.rng = random.Random(seed)

        self._templates = DOMAIN_TEMPLATES
        self._scenarios: List[Scenario] = []
        self._load_or_build()

        logger.info(
            f"ScenarioGenerator ready: {len(self._scenarios)} scenarios "
            f"across {len(set(s.domain for s in self._scenarios))} domains."
        )

    # ── public API ───────────────────────────────────────────────────────

    def sample(
        self,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Scenario:
        """Sample one scenario, optionally filtered."""
        pool = self._filter(domain, difficulty)
        if not pool:
            raise ValueError(
                f"No scenarios match domain={domain}, difficulty={difficulty}. "
                f"Available domains: {self.available_domains()}"
            )
        scenario = self.rng.choice(pool)
        # Reshuffle which reqs are hidden each call — adds training variance
        return self._apply_hiding(scenario)

    def sample_batch(
        self,
        n: int,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[Scenario]:
        """Sample n scenarios (with replacement)."""
        return [self.sample(domain=domain, difficulty=difficulty) for _ in range(n)]

    def available_domains(self) -> List[str]:
        return sorted(set(s.domain for s in self._scenarios))

    def available_difficulties(self) -> List[str]:
        return sorted(set(s.difficulty for s in self._scenarios))

    def stats(self) -> dict:
        """Summary statistics for logging."""
        from collections import Counter
        return {
            "total": len(self._scenarios),
            "by_domain": dict(Counter(s.domain for s in self._scenarios)),
            "by_difficulty": dict(Counter(s.difficulty for s in self._scenarios)),
            "avg_reqs": sum(len(s.ground_truth_reqs) for s in self._scenarios) / len(self._scenarios),
            "avg_hidden": sum(len(s.hidden_reqs) for s in self._scenarios) / len(self._scenarios),
        }

    # ── internal helpers ─────────────────────────────────────────────────

    def _load_or_build(self):
        cache_file = self.scenario_dir / "all_scenarios.json"
        if cache_file.exists():
            with open(cache_file) as f:
                raw = json.load(f)
            self._scenarios = [Scenario.from_dict(d) for d in raw]
            logger.info(f"Loaded {len(self._scenarios)} cached scenarios.")
        else:
            self._build_and_cache(cache_file)

    def _build_and_cache(self, cache_file: pathlib.Path):
        """Convert templates → Scenario objects and persist."""
        for t in self._templates:
            scenario = self._template_to_scenario(t)
            self._scenarios.append(scenario)

        with open(cache_file, "w") as f:
            json.dump([s.to_dict() for s in self._scenarios], f, indent=2)
        logger.info(f"Built and cached {len(self._scenarios)} scenarios.")

    def _template_to_scenario(self, t: dict) -> Scenario:
        reqs = t["ground_truth_reqs"]
        n_hide = max(1, int(len(reqs) * self.hide_fraction))
        hidden = self.rng.sample(reqs, n_hide)
        visible = [r for r in reqs if r not in hidden]

        # Deterministic scenario ID from content hash
        content = t["domain"] + t["rough_idea"]
        sid = hashlib.md5(content.encode()).hexdigest()[:8]

        stakeholders = t.get("stakeholders", [])
        # Convert to Stakeholder objects if they're dicts
        if stakeholders and isinstance(stakeholders[0], dict):
            stakeholders = [Stakeholder(**s) for s in stakeholders]

        return Scenario(
            scenario_id=sid,
            domain=t["domain"],
            rough_idea=t["rough_idea"],
            ground_truth_reqs=reqs,
            hidden_reqs=hidden,
            visible_reqs=visible,
            nfr=t.get("nfr", []),
            stakeholders=stakeholders,
            domain_entities=t.get("domain_entities", []),
            conflicts=t.get("conflicts", []),
            difficulty=t.get("difficulty", "medium"),
        )

    def _apply_hiding(self, scenario: Scenario) -> Scenario:
        """Re-randomise which reqs are hidden (new each call)."""
        reqs = scenario.ground_truth_reqs
        n_hide = max(1, int(len(reqs) * self.hide_fraction))
        hidden = self.rng.sample(reqs, n_hide)
        visible = [r for r in reqs if r not in hidden]
        # Return a new Scenario with updated hidden/visible split
        import copy
        s = copy.deepcopy(scenario)
        s.hidden_reqs = hidden
        s.visible_reqs = visible
        return s

    def _filter(
        self,
        domain: Optional[str],
        difficulty: Optional[str],
    ) -> List[Scenario]:
        pool = self._scenarios
        if domain:
            pool = [s for s in pool if s.domain == domain]
        if difficulty:
            pool = [s for s in pool if s.difficulty == difficulty]
        return pool


# ─────────────────────────────────────────────
#  Quick smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gen = ScenarioGenerator("data/scenarios/")
    print("\n── Stats ──")
    import pprint; pprint.pprint(gen.stats())

    print("\n── Sample scenario ──")
    s = gen.sample()
    print(f"Domain   : {s.domain}")
    print(f"Idea     : {s.rough_idea[:80]}...")
    print(f"Total FR : {len(s.ground_truth_reqs)}")
    print(f"Visible  : {len(s.visible_reqs)}  |  Hidden: {len(s.hidden_reqs)}")
    print(f"NFR      : {len(s.nfr)}")
    print(f"Conflict : {len(s.conflicts)}")
    print(f"Difficulty: {s.difficulty}")
    print(f"\nHidden reqs the agents must discover:")
    for r in s.hidden_reqs:
        print(f"  - {r}")
