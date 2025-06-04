import openai
import json
import datetime
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data models
@dataclass
class Patient:
    id: str
    name: str
    age: int
    medical_history: List[str]
    current_medications: List[str]
    vital_signs: Dict[str, float]
    risk_factors: List[str]

@dataclass
class Appointment:
    id: str
    patient_id: str
    doctor_id: str
    datetime: datetime.datetime
    type: str
    status: str
    notes: Optional[str] = None

@dataclass
class DrugCandidate:
    name: str
    mechanism: str
    target_disease: str
    safety_score: float
    efficacy_score: float
    development_stage: str

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PatientAlert:
    patient_id: str
    alert_type: str
    level: AlertLevel
    message: str
    timestamp: datetime.datetime
    recommended_action: str

class HealthcareAI:
    """Main Agentic AI system for healthcare applications"""
    
    def __init__(self, api_key: str = None):
        # Get API key from environment variable or parameter
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.client = openai.OpenAI(api_key=api_key)
        self.patients = {}
        self.appointments = {}
        self.alerts = []
        
    async def process_natural_language_request(self, request: str, context: Dict = None) -> Dict:
        """Process natural language requests and route to appropriate agents"""
        
        system_prompt = """
        You are a healthcare AI agent coordinator. Analyze the user request and determine which healthcare agent should handle it:
        
        1. APPOINTMENT_SCHEDULING - for booking, rescheduling, or managing appointments
        2. DRUG_DISCOVERY - for drug research, compound analysis, or treatment recommendations
        3. PATIENT_MONITORING - for vital signs analysis, risk assessment, or health alerts
        4. GENERAL_QUERY - for general healthcare information
        
        Respond with JSON format:
        {
            "agent_type": "AGENT_NAME",
            "intent": "specific intent",
            "parameters": {extracted parameters},
            "priority": "low|medium|high|critical"
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Request: {request}\nContext: {json.dumps(context or {})}"}
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Routing request to {result['agent_type']} agent")
            
            # Route to appropriate agent
            if result["agent_type"] == "APPOINTMENT_SCHEDULING":
                return await self.appointment_agent(request, result["parameters"])
            elif result["agent_type"] == "DRUG_DISCOVERY":
                return await self.drug_discovery_agent(request, result["parameters"])
            elif result["agent_type"] == "PATIENT_MONITORING":
                return await self.patient_monitoring_agent(request, result["parameters"])
            else:
                return await self.general_healthcare_agent(request, result["parameters"])
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e), "status": "failed"}

    async def appointment_agent(self, request: str, parameters: Dict) -> Dict:
        """Agent for handling appointment scheduling and management"""
        
        system_prompt = """
        You are an appointment scheduling AI agent. You can:
        - Schedule new appointments
        - Reschedule existing appointments
        - Cancel appointments
        - Check availability
        - Send reminders
        
        Consider patient preferences, doctor availability, urgency, and medical requirements.
        Always prioritize patient safety and care continuity.
        """
        
        try:
            # Simulate appointment scheduling logic
            if "schedule" in request.lower():
                appointment = await self._schedule_appointment(parameters)
                return {
                    "action": "appointment_scheduled",
                    "appointment": appointment.__dict__ if appointment else None,
                    "message": "Appointment scheduled successfully",
                    "status": "success"
                }
            elif "reschedule" in request.lower():
                result = await self._reschedule_appointment(parameters)
                return result
            elif "cancel" in request.lower():
                result = await self._cancel_appointment(parameters)
                return result
            else:
                # Use GPT-4 for complex scheduling decisions
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request}
                    ],
                    temperature=0.3
                )
                
                return {
                    "action": "appointment_consultation",
                    "recommendation": response.choices[0].message.content,
                    "status": "success"
                }
                
        except Exception as e:
            logger.error(f"Appointment agent error: {e}")
            return {"error": str(e), "status": "failed"}

    async def drug_discovery_agent(self, request: str, parameters: Dict) -> Dict:
        """Agent for drug discovery and treatment recommendations"""
        
        system_prompt = """
        You are a drug discovery AI agent with expertise in:
        - Molecular analysis and drug-target interactions
        - Safety and efficacy assessment
        - Treatment protocol recommendations
        - Drug repurposing opportunities
        - Clinical trial design suggestions
        
        Always emphasize safety, evidence-based recommendations, and regulatory compliance.
        Never provide medical advice for individual patients without proper clinical oversight.
        """
        
        try:
            # Analyze drug discovery request
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this drug discovery request: {request}"}
                ],
                temperature=0.2
            )
            
            # Simulate drug candidate analysis
            if "analyze compound" in request.lower():
                candidates = await self._analyze_drug_candidates(parameters)
                return {
                    "action": "compound_analysis",
                    "candidates": [c.__dict__ for c in candidates],
                    "analysis": response.choices[0].message.content,
                    "status": "success"
                }
            elif "treatment recommendation" in request.lower():
                recommendation = await self._generate_treatment_recommendation(parameters)
                return {
                    "action": "treatment_recommendation",
                    "recommendation": recommendation,
                    "analysis": response.choices[0].message.content,
                    "status": "success"
                }
            else:
                return {
                    "action": "drug_discovery_consultation",
                    "analysis": response.choices[0].message.content,
                    "status": "success"
                }
                
        except Exception as e:
            logger.error(f"Drug discovery agent error: {e}")
            return {"error": str(e), "status": "failed"}

    async def patient_monitoring_agent(self, request: str, parameters: Dict) -> Dict:
        """Agent for patient monitoring and health alerts"""
        
        system_prompt = """
        You are a patient monitoring AI agent responsible for:
        - Analyzing patient vital signs and health data
        - Detecting anomalies and health risks
        - Generating alerts for medical staff
        - Recommending interventions
        - Tracking patient progress
        
        Prioritize patient safety and early intervention. Generate appropriate alerts based on clinical guidelines.
        """
        
        try:
            # Analyze patient data
            if "monitor patient" in request.lower():
                alerts = await self._monitor_patient_vitals(parameters)
                return {
                    "action": "patient_monitoring",
                    "alerts": [alert.__dict__ for alert in alerts],
                    "status": "success"
                }
            elif "risk assessment" in request.lower():
                risk_analysis = await self._assess_patient_risk(parameters)
                return {
                    "action": "risk_assessment",
                    "analysis": risk_analysis,
                    "status": "success"
                }
            else:
                # Use GPT-4 for complex monitoring analysis
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this patient monitoring scenario: {request}"}
                    ],
                    temperature=0.1
                )
                
                return {
                    "action": "monitoring_analysis",
                    "analysis": response.choices[0].message.content,
                    "status": "success"
                }
                
        except Exception as e:
            logger.error(f"Patient monitoring agent error: {e}")
            return {"error": str(e), "status": "failed"}

    async def general_healthcare_agent(self, request: str, parameters: Dict) -> Dict:
        """General healthcare information and consultation agent"""
        
        system_prompt = """
        You are a general healthcare AI assistant providing:
        - Medical information and education
        - Healthcare guidance and best practices
        - Clinical decision support
        - Healthcare system navigation
        
        Always provide evidence-based information and emphasize the importance of professional medical consultation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request}
                ],
                temperature=0.3
            )
            
            return {
                "action": "general_consultation",
                "response": response.choices[0].message.content,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"General healthcare agent error: {e}")
            return {"error": str(e), "status": "failed"}

    # Helper methods for each agent
    async def _schedule_appointment(self, parameters: Dict) -> Optional[Appointment]:
        """Schedule a new appointment"""
        try:
            appointment = Appointment(
                id=f"apt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                patient_id=parameters.get("patient_id", "unknown"),
                doctor_id=parameters.get("doctor_id", "available"),
                datetime=datetime.datetime.now() + datetime.timedelta(days=7),
                type=parameters.get("appointment_type", "consultation"),
                status="scheduled"
            )
            self.appointments[appointment.id] = appointment
            return appointment
        except Exception as e:
            logger.error(f"Error scheduling appointment: {e}")
            return None

    async def _reschedule_appointment(self, parameters: Dict) -> Dict:
        """Reschedule an existing appointment"""
        appointment_id = parameters.get("appointment_id")
        if appointment_id in self.appointments:
            self.appointments[appointment_id].datetime = datetime.datetime.now() + datetime.timedelta(days=14)
            return {"message": "Appointment rescheduled successfully", "status": "success"}
        return {"message": "Appointment not found", "status": "failed"}

    async def _cancel_appointment(self, parameters: Dict) -> Dict:
        """Cancel an appointment"""
        appointment_id = parameters.get("appointment_id")
        if appointment_id in self.appointments:
            self.appointments[appointment_id].status = "cancelled"
            return {"message": "Appointment cancelled successfully", "status": "success"}
        return {"message": "Appointment not found", "status": "failed"}

    async def _analyze_drug_candidates(self, parameters: Dict) -> List[DrugCandidate]:
        """Analyze drug candidates for a specific condition"""
        # Simulate drug candidate analysis
        candidates = [
            DrugCandidate(
                name="Compound-A123",
                mechanism="Selective inhibitor",
                target_disease=parameters.get("condition", "Unknown"),
                safety_score=8.5,
                efficacy_score=7.2,
                development_stage="Phase II"
            ),
            DrugCandidate(
                name="BioMol-X456",
                mechanism="Receptor agonist",
                target_disease=parameters.get("condition", "Unknown"),
                safety_score=7.8,
                efficacy_score=8.1,
                development_stage="Preclinical"
            )
        ]
        return candidates

    async def _generate_treatment_recommendation(self, parameters: Dict) -> Dict:
        """Generate personalized treatment recommendations"""
        return {
            "primary_treatment": "Evidence-based therapy protocol",
            "alternative_options": ["Option A", "Option B"],
            "monitoring_requirements": ["Weekly lab work", "Monthly check-ups"],
            "contraindications": parameters.get("contraindications", []),
            "expected_outcomes": "Positive response expected in 4-6 weeks"
        }

    async def _monitor_patient_vitals(self, parameters: Dict) -> List[PatientAlert]:
        """Monitor patient vital signs and generate alerts"""
        alerts = []
        
        # Simulate vital signs analysis
        if parameters.get("heart_rate", 70) > 100:
            alerts.append(PatientAlert(
                patient_id=parameters.get("patient_id", "unknown"),
                alert_type="Vital Signs",
                level=AlertLevel.MEDIUM,
                message="Elevated heart rate detected",
                timestamp=datetime.datetime.now(),
                recommended_action="Monitor closely, consider cardiology consultation"
            ))
        
        if parameters.get("blood_pressure_systolic", 120) > 140:
            alerts.append(PatientAlert(
                patient_id=parameters.get("patient_id", "unknown"),
                alert_type="Blood Pressure",
                level=AlertLevel.HIGH,
                message="Hypertension detected",
                timestamp=datetime.datetime.now(),
                recommended_action="Immediate medical evaluation required"
            ))
        
        return alerts

    async def _assess_patient_risk(self, parameters: Dict) -> Dict:
        """Assess patient risk factors"""
        risk_factors = parameters.get("risk_factors", [])
        age = parameters.get("age", 50)
        
        risk_score = len(risk_factors) * 2 + (age - 50) * 0.1
        risk_level = "low" if risk_score < 5 else "medium" if risk_score < 10 else "high"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "contributing_factors": risk_factors,
            "recommendations": [
                "Regular monitoring",
                "Lifestyle modifications",
                "Preventive care protocols"
            ]
        }

    def add_patient(self, patient: Patient):
        """Add a new patient to the system"""
        self.patients[patient.id] = patient
        logger.info(f"Added patient: {patient.name}")

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Retrieve patient information"""
        return self.patients.get(patient_id)

    def get_alerts(self, patient_id: str = None) -> List[PatientAlert]:
        """Get alerts for a specific patient or all alerts"""
        if patient_id:
            return [alert for alert in self.alerts if alert.patient_id == patient_id]
        return self.alerts

# Example usage and demonstration
async def main():
    """Demonstrate the Healthcare AI system"""
    
    # Initialize the AI system using environment variable
    try:
        ai_system = HealthcareAI()  # Will automatically use OPENAI_API_KEY env var
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("Healthcare Agentic AI System Demo")
    print("=" * 50)
    
    # Example patient data
    patient = Patient(
        id="P001",
        name="John Doe",
        age=65,
        medical_history=["Hypertension", "Diabetes Type 2"],
        current_medications=["Metformin", "Lisinopril"],
        vital_signs={"heart_rate": 85, "blood_pressure": "140/90", "temperature": 98.6},
        risk_factors=["Smoking history", "Family history of heart disease"]
    )
    
    print(f"Patient: {patient.name}")
    print(f"Age: {patient.age}")
    print(f"Medical History: {', '.join(patient.medical_history)}")
    print("\nDemo Scenarios:")
    print("1. Appointment Scheduling")
    print("2. Drug Discovery Analysis")
    print("3. Patient Monitoring")
    
    print("\nTo use this system:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Install required packages: pip install -r requirements.txt")
    print("3. Initialize the system: ai = HealthcareAI()")
    print("4. Process requests: await ai.process_natural_language_request('Schedule appointment for patient')")

if __name__ == "__main__":
    asyncio.run(main())