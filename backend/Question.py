import json

with open('question_data.json', 'r') as f:
    questions = json.load(f)

# print(json.dumps(questions))


def find_question(label):
    return questions[label]