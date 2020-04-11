import numpy as np
import json

__all__ = ['Question', 'Program']

class Question(object):
    def __init__(self, json):
        super().__init__()
        self.img_index = json['image_index']
        self.img_file = json['image_filename']
        self.question_index = json['question_index']
        self.question_family = json['question_family_index']
        self.split = json['split']
        self.answer = json['answer']
        self.question = json['question']
        self.program = Question.get_program(json['program'])

    @staticmethod
    def get_program(programs_json):
        return [Program(p) for p in programs_json]

class Program(object):
    def __init__(self, json):
        super().__init__()
        self.inputs = json['inputs']
        self.function = json['function'] if 'function' in json else json['type']
        self.values = json['value_inputs']
        self.output = json['_output'] if '_output' in json else []