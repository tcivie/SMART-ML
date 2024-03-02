
import json

filename = input("Enter json path:")
with open(filename, mode='r') as fp:
    json_data = json.load(fp)

logics = json_data['logics']
csv_text = 'time,tls_id,duration,'
added_headers = False

runtime = json_data.get('runtime')
for t in runtime:
    data = runtime[t]
    for t_id in data:
        tls = data[t_id]
        if not added_headers:
            csv_text += ",".join(f"{key}" for key in tls) + "\n"
            added_headers = True
        if sum(tls.values()) == 0:
            continue
        phase_index = tls['current_phase_index']
        csv_text += f"{t},{t_id},{logics[t_id][phase_index]['duration']}," + ",".join(
            str(v) for v in tls.values()) + "\n"

with open(filename.replace(".json", ".csv"), mode='w') as fp:
    fp.write(csv_text)
