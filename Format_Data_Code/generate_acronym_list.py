import json

if __name__ == "__main__":
    csv = []
    with open("data\cleaned_data.csv") as fl:
        fl.readline()
        for line in fl:
            res = line.replace("\n", "").split(',')
            csv.append(res)
    rtn = {}
    for i in csv:
        for j in [4, 5]:
            if i[j] not in rtn:
                rtn[i[j]] = ""
    
    with open("data\\acronym_to_team.json", "w") as fl:
        json.dump(rtn, fl)
