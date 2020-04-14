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
        self.program = Question.get_program_seq(json['program'])

    @staticmethod
    def get_program_seq(programs_json):
        return [Program(p) for p in programs_json]

class Program(object):
    def __init__(self, json):
        super().__init__()
        inputs = json['inputs']
        function = json['function'] if 'function' in json else json['type']
        value_inputs = json['value_inputs']

        op = get_function_operation(function)
        if op == 'scene':
            self.operator = 'scene'
        elif op.startswith('filter'):
            self.operator = 'filter'
            self.concept = value_inputs[0]
        elif op.startswith('relate'):
            self.operator = 'relate'
            self.relational_concept = value_inputs[0]
        elif op.startswith('same'):
            self.operator = 'relate_attribute_equal'
            self.attribute = get_function_attribute(function)
        elif op in ('intersect', 'union'):
            self.operator = op
        elif op == 'unique':
            pass  # We will ignore the unique operations.
        else:
            if op.startswith('query'):
                self.operator = 'query'
                self.attribute = get_function_attribute(function)
            elif op.startswith('equal') and op != 'equal_integer':
                self.operator = 'query_attribute_equal'
                self.attribute = get_function_attribute(function)
            elif op == 'exist':
                self.operator = 'exist'
            elif op == 'count':
                self.operator = 'count'
            elif op == 'equal_integer':
                self.operator = 'count_equal'
            elif op == 'less_than':
                self.operator = 'count_less'
            elif op == 'greater_than':
                self.operator = 'count_greater'
            else:
                raise ValueError('Unknown CLEVR operation: {}.'.format(op))

def get_function_operation(function):
        return function.split('_')[0]
    
def get_function_attribute(function):
    return function.split('_')[1]
