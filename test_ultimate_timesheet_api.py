
"""
Comprehensive Test Suite for Ultimate Expert Timesheet API
Tests all endpoints and functionality with professional coverage
"""

import pytest
import requests
import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Any
import asyncio
import httpx

# Test Configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER_EMAIL = "test.user@company.com"
TEST_USER_EMAIL_2 = "demo.user@company.com"

class TimesheetAPITester:
    """Professional test suite for the Ultimate Timesheet API"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def test_api_health(self) -> Dict[str, Any]:
        """Test API health endpoint"""
        print("🔍 Testing API Health...")

        try:
            response = self.session.get(f"{self.base_url}/health")

            result = {
                "endpoint": "GET /health",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_time": response.elapsed.total_seconds(),
                "data": response.json() if response.status_code == 200 else None
            }

            print(f"✅ Health Check: {result['success']} (Status: {result['status_code']})")
            if result['data']:
                print(f"   Version: {result['data'].get('version', 'Unknown')}")
                print(f"   Status: {result['data'].get('status', 'Unknown')}")

            return result

        except Exception as e:
            print(f"❌ Health Check Failed: {str(e)}")
            return {"endpoint": "GET /health", "success": False, "error": str(e)}

    def test_root_endpoint(self) -> Dict[str, Any]:
        """Test root endpoint"""
        print("\n🏠 Testing Root Endpoint...")

        try:
            response = self.session.get(f"{self.base_url}/")

            result = {
                "endpoint": "GET /",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_time": response.elapsed.total_seconds(),
                "data": response.json() if response.status_code == 200 else None
            }

            print(f"✅ Root Endpoint: {result['success']} (Status: {result['status_code']})")
            if result['data']:
                print(f"   Message: {result['data'].get('message', 'No message')}")
                print(f"   Expertise: {result['data'].get('expertise', 'Unknown')}")

            return result

        except Exception as e:
            print(f"❌ Root Endpoint Failed: {str(e)}")
            return {"endpoint": "GET /", "success": False, "error": str(e)}

    def test_chat_conversation(self, test_scenarios: List[Dict]) -> List[Dict[str, Any]]:
        """Test conversational chat functionality"""
        print("\n💬 Testing Chat Conversations...")

        results = []

        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n   Scenario {i}: {scenario['description']}")

            try:
                payload = {
                    "email": scenario.get("email", TEST_USER_EMAIL),
                    "user_prompt": scenario["prompt"]
                }

                response = self.session.post(f"{self.base_url}/chat", json=payload)

                result = {
                    "scenario": scenario['description'],
                    "prompt": scenario["prompt"],
                    "endpoint": "POST /chat",
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time": response.elapsed.total_seconds(),
                    "data": response.json() if response.status_code == 200 else None
                }

                if result['success'] and result['data']:
                    print(f"   ✅ Success: {result['data']['conversation_phase']}")
                    print(f"   Response: {result['data']['response'][:100]}...")
                    if result['data'].get('tabular_data'):
                        print("   📊 Tabular data included")
                else:
                    print(f"   ❌ Failed: Status {result['status_code']}")

                results.append(result)

                # Brief pause between requests
                time.sleep(0.5)

            except Exception as e:
                print(f"   ❌ Exception: {str(e)}")
                results.append({
                    "scenario": scenario['description'],
                    "success": False,
                    "error": str(e)
                })

        return results

    def test_project_codes(self) -> Dict[str, Any]:
        """Test project codes endpoints"""
        print("\n📋 Testing Project Codes...")

        results = {}

        # Test all projects
        try:
            response = self.session.get(f"{self.base_url}/projects")
            results["all_projects"] = {
                "endpoint": "GET /projects",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ All Projects: {results['all_projects']['success']}")

        except Exception as e:
            results["all_projects"] = {"success": False, "error": str(e)}
            print(f"   ❌ All Projects Failed: {str(e)}")

        # Test Oracle projects
        for system in ["Oracle", "Mars"]:
            try:
                response = self.session.get(f"{self.base_url}/projects/{system}")
                results[f"{system.lower()}_projects"] = {
                    "endpoint": f"GET /projects/{system}",
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "data": response.json() if response.status_code == 200 else None
                }

                if results[f"{system.lower()}_projects"]['success']:
                    data = results[f"{system.lower()}_projects"]['data']
                    project_count = data.get('count', 0)
                    print(f"   ✅ {system} Projects: {project_count} projects found")
                else:
                    print(f"   ❌ {system} Projects Failed")

            except Exception as e:
                results[f"{system.lower()}_projects"] = {"success": False, "error": str(e)}
                print(f"   ❌ {system} Projects Exception: {str(e)}")

        return results

    def test_timesheet_operations(self) -> Dict[str, Any]:
        """Test timesheet view operations"""
        print("\n📊 Testing Timesheet Operations...")

        results = {}

        for system in ["Oracle", "Mars"]:
            try:
                # Test basic timesheet retrieval
                response = self.session.get(f"{self.base_url}/timesheet/{TEST_USER_EMAIL}/{system}")

                results[f"{system.lower()}_timesheet"] = {
                    "endpoint": f"GET /timesheet/{TEST_USER_EMAIL}/{system}",
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "data": response.json() if response.status_code == 200 else None
                }

                if results[f"{system.lower()}_timesheet"]['success']:
                    data = results[f"{system.lower()}_timesheet"]['data']
                    entry_count = data.get('count', 0)
                    total_hours = data.get('total_hours', 0)
                    print(f"   ✅ {system} Timesheet: {entry_count} entries, {total_hours} hours")
                else:
                    print(f"   ⚠️ {system} Timesheet: No entries or error")

            except Exception as e:
                results[f"{system.lower()}_timesheet"] = {"success": False, "error": str(e)}
                print(f"   ❌ {system} Timesheet Exception: {str(e)}")

        # Test with date filtering
        try:
            start_date = (date.today() - timedelta(days=30)).isoformat()
            end_date = date.today().isoformat()

            response = self.session.get(
                f"{self.base_url}/timesheet/{TEST_USER_EMAIL}/Oracle",
                params={"start_date": start_date, "end_date": end_date}
            )

            results["date_filtered"] = {
                "endpoint": "GET /timesheet with date filter",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }

            print(f"   ✅ Date Filtered: {results['date_filtered']['success']}")

        except Exception as e:
            results["date_filtered"] = {"success": False, "error": str(e)}
            print(f"   ❌ Date Filtered Exception: {str(e)}")

        return results

    def test_timesheet_submission(self) -> Dict[str, Any]:
        """Test timesheet entry submission"""
        print("\n📝 Testing Timesheet Submission...")

        # Test data
        test_entries = [
            {
                "system": "Oracle",
                "date": date.today().isoformat(),
                "hours": 8.0,
                "project_code": "ORG-001",
                "task_code": "TEST-001",
                "comments": "API Testing - Oracle Entry"
            },
            {
                "system": "Mars", 
                "date": date.today().isoformat(),
                "hours": 4.0,
                "project_code": "MRS-001",
                "task_code": "TEST-002",
                "comments": "API Testing - Mars Entry"
            }
        ]

        try:
            response = self.session.post(
                f"{self.base_url}/timesheet/submit",
                params={"user_email": TEST_USER_EMAIL},
                json=test_entries
            )

            result = {
                "endpoint": "POST /timesheet/submit",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }

            if result['success']:
                data = result['data']
                print(f"   ✅ Submission Success: {data.get('entries_submitted', 0)} entries")
                print(f"   Total Hours: {data.get('total_hours', 0)}")
            else:
                print(f"   ❌ Submission Failed: Status {result['status_code']}")

            return result

        except Exception as e:
            print(f"   ❌ Submission Exception: {str(e)}")
            return {"success": False, "error": str(e)}

    def test_draft_operations(self) -> Dict[str, Any]:
        """Test draft timesheet operations"""
        print("\n📄 Testing Draft Operations...")

        # Test draft save
        draft_entries = [
            {
                "system": "Oracle",
                "date": date.today().isoformat(),
                "hours": 6.0,
                "project_code": "ORG-002",
                "comments": "Draft entry for testing"
            }
        ]

        try:
            response = self.session.post(
                f"{self.base_url}/timesheet/draft",
                params={"user_email": TEST_USER_EMAIL},
                json=draft_entries
            )

            result = {
                "endpoint": "POST /timesheet/draft",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }

            if result['success']:
                data = result['data']
                print(f"   ✅ Draft Saved: {data.get('draft_id', 'Unknown ID')}")
                print(f"   Total Hours: {data.get('total_hours', 0)}")
            else:
                print(f"   ❌ Draft Save Failed: Status {result['status_code']}")

            return result

        except Exception as e:
            print(f"   ❌ Draft Exception: {str(e)}")
            return {"success": False, "error": str(e)}

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🚀 STARTING COMPREHENSIVE API TESTS")
        print("=" * 60)

        start_time = time.time()

        # Test scenarios for conversation
        chat_scenarios = [
            {
                "description": "Help command",
                "prompt": "help"
            },
            {
                "description": "Show Oracle projects",
                "prompt": "show projects Oracle"
            },
            {
                "description": "Show Mars projects", 
                "prompt": "show projects Mars"
            },
            {
                "description": "Show Oracle timesheet",
                "prompt": "show timesheet Oracle"
            },
            {
                "description": "Show Mars timesheet",
                "prompt": "show timesheet Mars"
            },
            {
                "description": "Start fresh",
                "prompt": "start fresh"
            },
            {
                "description": "Single system entry",
                "prompt": "8 hours on Oracle project ORG-001 yesterday"
            },
            {
                "description": "Multi-system entry",
                "prompt": "Oracle: 4 hours ORG-001, Mars: 4 hours MRS-001, both today"
            },
            {
                "description": "Timesheet with task and comments",
                "prompt": "6 hours Oracle ORG-002 today, task DEV-001, comments: Testing API functionality"
            }
        ]

        # Run all tests
        results = {
            "test_start_time": datetime.utcnow().isoformat(),
            "api_health": self.test_api_health(),
            "root_endpoint": self.test_root_endpoint(),
            "project_codes": self.test_project_codes(),
            "timesheet_operations": self.test_timesheet_operations(),
            "chat_conversations": self.test_chat_conversation(chat_scenarios),
            "timesheet_submission": self.test_timesheet_submission(),
            "draft_operations": self.test_draft_operations()
        }

        end_time = time.time()
        total_time = end_time - start_time

        results["test_end_time"] = datetime.utcnow().isoformat()
        results["total_test_time"] = total_time

        # Generate summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        success_count = 0
        total_tests = 0

        for test_category, test_results in results.items():
            if test_category.startswith("test_"):
                continue

            if isinstance(test_results, dict):
                if test_results.get('success'):
                    success_count += 1
                total_tests += 1
            elif isinstance(test_results, list):
                for result in test_results:
                    if result.get('success'):
                        success_count += 1
                    total_tests += 1

        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0

        print(f"✅ Successful Tests: {success_count}/{total_tests} ({success_rate:.1f}%)")
        print(f"⏱️ Total Test Time: {total_time:.2f} seconds")
        print(f"🎯 API Base URL: {self.base_url}")

        results["summary"] = {
            "success_count": success_count,
            "total_tests": total_tests,
            "success_rate": success_rate
        }

        return results

def main():
    """Main test execution"""
    print("🧪 Ultimate Expert Timesheet API - Test Suite")
    print("=" * 60)

    tester = TimesheetAPITester()
    results = tester.run_comprehensive_tests()

    # Save results to file
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Test results saved to: test_results.json")

    # Print key findings
    if results['summary']['success_rate'] >= 80:
        print("\n🎉 API IS READY FOR PRODUCTION!")
    elif results['summary']['success_rate'] >= 60:
        print("\n⚠️ API needs some improvements before production")
    else:
        print("\n❌ API requires significant fixes")

if __name__ == "__main__":
    main()
