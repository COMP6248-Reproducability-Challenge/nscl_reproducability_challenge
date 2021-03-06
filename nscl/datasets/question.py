import torch

from nscl.datasets.clevr_definition import CLEVRDefinition, QuestionTypes

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
        self.raw_question = json['question']
        self.program = Question.get_program_seq(json['program'])
        self.answer_tensor = Question.get_answer_tensor(self.answer)
        self.question_type = Question.get_question_type(self.answer)
        self.synthetic = False

    @staticmethod
    def get_program_seq(programs_json):
        return [Program(p) for p in programs_json]

    @staticmethod
    def get_answer_tensor(answer):
        if answer == 'yes':
            return torch.tensor([1., 0.], dtype=torch.float)
        if answer == 'no':
            return torch.tensor([0., 1.], dtype=torch.float)
        if answer.isdigit():
            return torch.tensor(float(answer), dtype=torch.float)

        for attr, concepts in CLEVRDefinition.attribute_concept_map.items():
            if answer in concepts:
                return torch.tensor([concepts.index(answer)], dtype=torch.long)

        raise Exception('Unknown answer')

    @staticmethod
    def get_question_type(answer):
        if answer == 'yes' or answer == 'no':
            return QuestionTypes.BOOLEAN
        if answer.isdigit():
            return QuestionTypes.COUNT

        return QuestionTypes.ATTRIBUTE

    @staticmethod
    def gen_empty_question():
        json = dict()
        json['image_index'] = None
        json['image_filename'] = None
        json['question_index'] = None
        json['question_family_index'] = None
        json['split'] = None
        json['answer'] = 'yes'
        json['question'] = None
        json['program'] = []
        q = Question(json)
        q.synthetic = True
        scene_program = Program.gen_scene_program()
        q.program.append(scene_program)
        return q

    @staticmethod
    def gen_count_question(concept):
        q = Question.gen_empty_question()
        q.raw_question = f'How many {concept} things are there?'
        filter_program = Program.gen_filter_program(CLEVRDefinition.concept_attribute_map[concept], concept)
        filter_program.input_ids.append(0)
        count_program = Program.gen_count_program()
        count_program.input_ids.append(1)
        q.program.extend([filter_program, count_program])
        return q

    @staticmethod
    def gen_exist_question(concept):
        q = Question.gen_empty_question()
        q.raw_question = f'Is there a {concept} object?'
        filter_program = Program.gen_filter_program(CLEVRDefinition.concept_attribute_map[concept], concept)
        filter_program.input_ids.append(0)
        exist_program = Program.gen_exist_program()
        exist_program.input_ids.append(1)
        q.program.extend([filter_program, exist_program])
        return q


class Program(object):
    def __init__(self, json):
        super().__init__()
        inputs = json['inputs']
        function = json['function'] if 'function' in json else json['type']
        input_ids = json['inputs']
        value_inputs = json['value_inputs']

        self.operator = ''
        self.attribute = ''
        self.concept = ''
        self.input_ids = input_ids

        if function == 'scene':
            self.operator = 'scene'
        elif function.startswith('filter'):
            self.operator = 'filter'
            self.attribute = get_function_attribute(function)
            self.concept = value_inputs[0]
        elif function.startswith('relate'):
            self.operator = 'relate'
            self.concept = value_inputs[0]
        elif function.startswith('same'):
            self.operator = 'relate_attribute_equal'
            self.attribute = get_function_attribute(function)
        elif function in ('intersect', 'union'):
            self.operator = function
        elif function == 'unique':
            self.operator = 'unique'
        else:
            if function.startswith('query'):
                self.operator = 'query'
                self.attribute = get_function_attribute(function)
            elif function.startswith('equal') and function != 'equal_integer':
                self.operator = 'query_attribute_equal'
                self.attribute = get_function_attribute(function)
            elif function == 'exist':
                self.operator = 'exist'
            elif function == 'count':
                self.operator = 'count'
            elif function == 'equal_integer':
                self.operator = 'count_equal'
            elif function == 'less_than':
                self.operator = 'count_less'
            elif function == 'greater_than':
                self.operator = 'count_greater'
            else:
                raise ValueError('Unknown CLEVR operation: {}.'.format(function))

    @staticmethod
    def gen_scene_program():
        json = dict()
        json['inputs'] = []
        json['function'] = 'scene'
        json['value_inputs'] = []
        return Program(json)

    @staticmethod
    def gen_count_program():
        json = dict()
        json['inputs'] = []
        json['function'] = 'count'
        json['value_inputs'] = []
        return Program(json)

    @staticmethod
    def gen_exist_program():
        json = dict()
        json['inputs'] = []
        json['function'] = 'exist'
        json['value_inputs'] = []
        return Program(json)

    @staticmethod
    def gen_filter_program(attr, concept):
        json = dict()
        json['inputs'] = []
        json['function'] = f'filter_{attr}'
        json['value_inputs'] = [concept]
        return Program(json)

def get_function_operation(function):
    return function.split('_')[0]


def get_function_attribute(function):
    return function.split('_')[1]
