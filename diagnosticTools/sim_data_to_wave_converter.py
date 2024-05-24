import json

filename = input("Enter json path:")
with open(filename, mode='r') as fp:
    json_data = json.load(fp)

logics = json_data['logics']
csv_text = 'time,' + ','.join((logics.keys())) + '\n'


def calculate_new_value(logic_dict: dict):
    return round(sum(logic_dict.values()) / len(logic_dict), 3)


runtime = json_data.get('runtime')
for t in runtime:
    data = runtime[t]
    csv_text += t + "," + ",".join(str(x) for x in map(calculate_new_value, data.values())) + '\n'

with open(filename.replace(".json", "_wave.csv"), mode='w') as fp:
    fp.write(csv_text)
