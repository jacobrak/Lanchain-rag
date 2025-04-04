import unittest
from models import ask_question_test

class Run_tests(unittest.TestCase):
    def test_question1(self):
        result = ask_question_test("Summarize 2013 annual performance report")
        self.assertNotEqual(result, 'Hallucination Detected')

    def test_question2(self):
        result = ask_question_test("What are the priorities in the 2022 fiscal year?")
        self.assertNotEqual(result, 'Hallucination Detected')

    def test_question3(self):
        result = ask_question_test("What is the HHS strategic plan?")
        self.assertNotEqual(result, 'Hallucination Detected')

    def test_question4(self):
        result = ask_question_test("What is GSA?")
        self.assertNotEqual(result, 'Hallucination Detected')

    def test_question5(self):
        result = ask_question_test("Give me an overview of OPM in 2018?")
        self.assertNotEqual(result, 'Hallucination Detected')
  