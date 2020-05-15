import unittest
import torch
import numpy as np
from nscl.datasets.question import Question

class QuestionTest(unittest.TestCase):
    def test_answer_tensor_bool(self):
        self.assertTrue(torch.tensor(1.0), Question.get_answer_tensor('yes'))
        self.assertTrue(torch.tensor(1.0), Question.get_answer_tensor('no'))

    def test_answer_tensor_count(self):
        self.assertTrue(torch.tensor(3.0), Question.get_answer_tensor('3'))
        self.assertTrue(torch.tensor(5.0), Question.get_answer_tensor('5'))

    def test_answer_tensor_concept(self):
        self.assertTrue((np.array([0., 1., 0., 0., 0., 0., 0., 0.]) == Question.get_answer_tensor('red').numpy()).all())
        self.assertTrue((np.array([0., 0., 0., 0., 0., 0., 0., 1.]) == Question.get_answer_tensor('yellow').numpy()).all())
        self.assertTrue((np.array([0., 0., 1., 0., 0., 0., 0., 0.]) == Question.get_answer_tensor('blue').numpy()).all())
        self.assertTrue((np.array([1., 0.]) == Question.get_answer_tensor('rubber').numpy()).all())
        self.assertTrue((np.array([0., 1.]) == Question.get_answer_tensor('metal').numpy()).all())
        self.assertTrue((np.array([0., 1., 0.]) == Question.get_answer_tensor('sphere').numpy()).all())
        self.assertTrue((np.array([1., 0., 0.]) == Question.get_answer_tensor('cube').numpy()).all())
