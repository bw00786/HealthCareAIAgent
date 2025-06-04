import asyncio
import json
from datetime import datetime, timedelta
import sys

# Import the healthcare AI system (assumes the previous code is saved as healthcare_ai.py)
# If you have it in the same file, you can skip this import
try:
    from healthcare_ai import HealthcareAI, Patient, AlertLevel
except ImportError:
    print("Please save the healthcare AI code as 'healthcare_ai.py' first")
    sys.exit(1)

class HealthcareAITester:
    def __init__(self, api_key: str):
        self.ai_system = HealthcareAI(api_key)
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Set up sample patients and data for testing"""
        
        # Sample patients
        patients = [
            Patient(
                id="P001",
                name="Alice Johnson",
                age=45,
                medical_history=["Hypertension", "High Cholesterol"],
                current_medications=["Lisinopril 10mg", "Atorvastatin 20mg"],
                vital_signs={"heart_rate": 75, "blood_pressure_systolic": 135, "blood_pressure_diastolic": 85, "temperature": 98.2},
                risk_factors=["Family history of heart disease", "Sedentary lifestyle"]
            ),
            Patient(
                id="P002",
                name="Robert Smith",
                age=68,
                medical_history=["Diabetes Type 2", "Chronic Kidney Disease"],
                current_medications=["Metformin 500mg", "Insulin"],
                vital_signs={"heart_rate": 88, "blood_pressure_systolic": 145, "blood_pressure_diastolic": 92, "temperature": 99.1},
                risk_factors=["Smoking history", "Diabetes complications"]
            ),
            Patient(
                id="P003",
                name="Maria Garcia",
                age=32,
                medical_history=["Asthma", "Allergies"],
                current_medications=["Albuterol inhaler", "Flonase"],
                vital_signs={"heart_rate": 72, "blood_pressure_systolic": 118, "blood_pressure_diastolic": 78, "temperature": 98.6},
                risk_factors=["Environmental allergies"]
            )
        ]
        
        # Add patients to the system
        for patient in patients:
            self.ai_system.add_patient(patient)
        
        print(f"✅ Added {len(patients)} sample patients to the system")

    async def test_appointment_scheduling(self):
        """Test appointment scheduling functionality"""
        print("\n" + "="*60)
        print("🗓️  TESTING APPOINTMENT SCHEDULING")
        print("="*60)
        
        test_requests = [
            "Schedule an appointment for patient P001 with cardiology next Tuesday",
            "I need to book a follow-up appointment for Alice Johnson with her primary care doctor",
            "Can you reschedule appointment apt_20241201_143000 to next week?",
            "What's the earliest available appointment for a diabetes consultation?"
        ]
        
        for i, request in enumerate(test_requests, 1):
            print(f"\n🧪 Test {i}: {request}")
            try:
                result = await self.ai_system.process_natural_language_request(
                    request, 
                    context={"patient_id": "P001", "preferred_time": "morning"}
                )
                print(f"✅ Result: {json.dumps(result, indent=2, default=str)}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    async def test_drug_discovery(self):
        """Test drug discovery functionality"""
        print("\n" + "="*60)
        print("💊 TESTING DRUG DISCOVERY")
        print("="*60)
        
        test_requests = [
            "Analyze potential drug compounds for treating hypertension",
            "What are the best treatment options for Type 2 diabetes in elderly patients?",
            "I need a safety analysis for compound XY-123 targeting cardiovascular disease",
            "Recommend alternative treatments for patients with kidney disease who can't take ACE inhibitors"
        ]
        
        for i, request in enumerate(test_requests, 1):
            print(f"\n🧪 Test {i}: {request}")
            try:
                result = await self.ai_system.process_natural_language_request(
                    request,
                    context={"condition": "hypertension", "patient_age": 65, "contraindications": ["kidney_disease"]}
                )
                print(f"✅ Result: {json.dumps(result, indent=2, default=str)}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    async def test_patient_monitoring(self):
        """Test patient monitoring functionality"""
        print("\n" + "="*60)
        print("📊 TESTING PATIENT MONITORING")
        print("="*60)
        
        test_requests = [
            "Monitor patient P002 - heart rate 105, blood pressure 160/95, temperature 100.2F",
            "Assess cardiovascular risk for patient Alice Johnson based on her current vitals",
            "Generate health alerts for patient with diabetes showing elevated glucose levels",
            "Analyze vital signs trend for patient P001 over the past week"
        ]
        
        # Test with different vital sign scenarios
        test_scenarios = [
            {
                "patient_id": "P002",
                "heart_rate": 105,
                "blood_pressure_systolic": 160,
                "blood_pressure_diastolic": 95,
                "temperature": 100.2,
                "glucose": 250
            },
            {
                "patient_id": "P001",
                "heart_rate": 72,
                "blood_pressure_systolic": 135,
                "blood_pressure_diastolic": 85,
                "temperature": 98.6
            }
        ]
        
        for i, request in enumerate(test_requests, 1):
            print(f"\n🧪 Test {i}: {request}")
            try:
                context = test_scenarios[i-1] if i <= len(test_scenarios) else {"patient_id": "P001"}
                result = await self.ai_system.process_natural_language_request(request, context=context)
                print(f"✅ Result: {json.dumps(result, indent=2, default=str)}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    async def test_general_healthcare(self):
        """Test general healthcare consultation"""
        print("\n" + "="*60)
        print("🏥 TESTING GENERAL HEALTHCARE CONSULTATION")
        print("="*60)
        
        test_requests = [
            "What are the best practices for managing diabetes in elderly patients?",
            "Explain the interaction between high blood pressure medications and kidney function",
            "What should I know about managing asthma triggers in adult patients?",
            "How do you assess cardiovascular risk in patients with multiple comorbidities?"
        ]
        
        for i, request in enumerate(test_requests, 1):
            print(f"\n🧪 Test {i}: {request}")
            try:
                result = await self.ai_system.process_natural_language_request(request)
                print(f"✅ Result: {json.dumps(result, indent=2, default=str)}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    async def test_complex_scenarios(self):
        """Test complex, multi-step healthcare scenarios"""
        print("\n" + "="*60)
        print("🔄 TESTING COMPLEX SCENARIOS")
        print("="*60)
        
        # Scenario 1: Emergency patient with multiple issues
        print("\n🚨 Scenario 1: Emergency Patient Management")
        emergency_request = """
        Patient P002 (Robert Smith, 68yo with diabetes and kidney disease) just arrived with:
        - Chest pain (7/10)
        - Heart rate: 115 bpm
        - Blood pressure: 180/110
        - Blood glucose: 300 mg/dL
        - Difficulty breathing
        
        What immediate actions should be taken and what specialists need to be consulted?
        """
        
        try:
            result = await self.ai_system.process_natural_language_request(
                emergency_request,
                context={
                    "patient_id": "P002",
                    "emergency": True,
                    "heart_rate": 115,
                    "blood_pressure_systolic": 180,
                    "blood_pressure_diastolic": 110,
                    "glucose": 300,
                    "pain_level": 7
                }
            )
            print(f"✅ Emergency Response: {json.dumps(result, indent=2, default=str)}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

        # Scenario 2: Treatment optimization
        print("\n💡 Scenario 2: Treatment Optimization")
        optimization_request = """
        Patient P001 (Alice Johnson) has been on current medications for 6 months.
        Recent labs show:
        - LDL cholesterol still elevated at 140 mg/dL
        - Blood pressure averaging 140/88
        - Patient reports muscle aches (possibly statin-related)
        
        Recommend treatment adjustments and monitoring plan.
        """
        
        try:
            result = await self.ai_system.process_natural_language_request(
                optimization_request,
                context={
                    "patient_id": "P001",
                    "ldl_cholesterol": 140,
                    "blood_pressure": "140/88",
                    "side_effects": ["muscle_aches"],
                    "current_medications": ["Lisinopril 10mg", "Atorvastatin 20mg"]
                }
            )
            print(f"✅ Optimization Plan: {json.dumps(result, indent=2, default=str)}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    def display_system_status(self):
        """Display current system status and data"""
        print("\n" + "="*60)
        print("📋 SYSTEM STATUS")
        print("="*60)
        
        print(f"👥 Patients in system: {len(self.ai_system.patients)}")
        for patient_id, patient in self.ai_system.patients.items():
            print(f"   - {patient.name} (ID: {patient_id}, Age: {patient.age})")
        
        print(f"📅 Appointments: {len(self.ai_system.appointments)}")
        print(f"🚨 Active alerts: {len(self.ai_system.alerts)}")
        
        # Display any alerts
        if self.ai_system.alerts:
            print("\n⚠️  Active Alerts:")
            for alert in self.ai_system.alerts:
                print(f"   - {alert.level.value.upper()}: {alert.message} (Patient: {alert.patient_id})")

    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting Healthcare AI System Tests")
        print("=" * 80)
        
        try:
            await self.test_appointment_scheduling()
            await self.test_drug_discovery()
            await self.test_patient_monitoring()
            await self.test_general_healthcare()
            await self.test_complex_scenarios()
            
            self.display_system_status()
            
            print("\n" + "="*80)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ TEST SUITE FAILED: {str(e)}")
            raise

async def main():
    """Main testing function"""
    print("Healthcare AI System - Interactive Testing")
    print("=" * 50)
    
    # Get API key from user
    api_key = input("Please enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ API key is required to run tests")
        return
    
    # Initialize tester
    try:
        tester = HealthcareAITester(api_key)
        
        print("\nChoose testing mode:")
        print("1. Run all tests automatically")
        print("2. Interactive testing (choose specific tests)")
        print("3. Custom request testing")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            await tester.run_all_tests()
        
        elif choice == "2":
            await interactive_testing(tester)
        
        elif choice == "3":
            await custom_request_testing(tester)
        
        else:
            print("Invalid choice. Running all tests...")
            await tester.run_all_tests()
            
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")
        print("Please check your API key and internet connection")

async def interactive_testing(tester):
    """Interactive testing mode"""
    while True:
        print("\n" + "="*50)
        print("Interactive Testing Menu:")
        print("1. Test Appointment Scheduling")
        print("2. Test Drug Discovery")
        print("3. Test Patient Monitoring")
        print("4. Test General Healthcare")
        print("5. Test Complex Scenarios")
        print("6. View System Status")
        print("7. Exit")
        
        choice = input("\nSelect test (1-7): ").strip()
        
        if choice == "1":
            await tester.test_appointment_scheduling()
        elif choice == "2":
            await tester.test_drug_discovery()
        elif choice == "3":
            await tester.test_patient_monitoring()
        elif choice == "4":
            await tester.test_general_healthcare()
        elif choice == "5":
            await tester.test_complex_scenarios()
        elif choice == "6":
            tester.display_system_status()
        elif choice == "7":
            break
        else:
            print("Invalid choice. Please try again.")

async def custom_request_testing(tester):
    """Custom request testing mode"""
    print("\n" + "="*50)
    print("Custom Request Testing")
    print("Enter your healthcare requests (type 'exit' to quit)")
    print("="*50)
    
    while True:
        request = input("\n🎯 Enter your request: ").strip()
        
        if request.lower() == 'exit':
            break
        
        if not request:
            continue
            
        try:
            print("🔄 Processing request...")
            result = await tester.ai_system.process_natural_language_request(request)
            print(f"\n✅ Response:")
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())