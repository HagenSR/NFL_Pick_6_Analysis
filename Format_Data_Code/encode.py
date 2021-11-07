import json

if __name__ == "__main__":
    mapping = {}
    csv = []
    with open("data\cleaned_data.csv") as fl:
        fl.readline()
        for line in fl:
            res = line.replace("\n", "").split(',')
            csv.append(res)

    for col in range(len(csv[0])):
        mapping[col] = {"count" : 0}
        for row in csv:
            if row[col] not in mapping[col]:
                mapping[col][row[col]] = mapping[col]["count"]
                mapping[col]["count"] += 1
    rtn = ""
    for row in csv:
        for col in range(len(row)):
            if col != len(row):
                rtn += "{0},".format(mapping[col][row[col]])
            else:
                rtn += "{0}".format(mapping[col][row[col]])
        rtn += "\n"

    with open("data\mapping.json","w") as fl:
        json.dump(mapping, fl)
    
    with open("data\csv.csv", "w") as fl:
        fl.write(rtn)



