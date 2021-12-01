import json
from sklearn import preprocessing

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    mapping = {}
    csv = []
    header = ""
    scoreBins = {"count": 13, "-22" : 1 , "-15 | -21" : 2, "-10 | -14" : 3, "-7 | -9" : 4, "-5 | -6" : 5, "-3 | -4" : 6, "-1 | -2" : 7, 
                  "1 | 2": 8 , "3 | 4" : 9, "5 | 6" : 10, "7 | 9" : 11, "10 | 14": 12, "15 | 21": 13 ,"22+" : 14}

    with open("data\cleaned_data.csv") as fl:
        header = fl.readline()
        for line in fl:
            res = line.replace("\n", "").split(',')
            csv.append(res)

    for col in range(len(csv[0][0:-1])):
        mapping[col] = {"count" : 0}
        for row in csv:
            if row[col] not in mapping[col] and not (row[col].isdigit() or isfloat(row[col])):
                mapping[col][row[col]] = mapping[col]["count"]
                mapping[col]["count"] += 1
            elif row[col] not in mapping[col]:
                mapping[col][row[col]] = row[col]
    mapping[len(csv[0]) -1] = scoreBins
    rtn = header 
    for row in csv:
        for col in range(len(row)):
            if col != len(row) -1:
                rtn += "{0},".format(mapping[col][row[col]])
            else:
                rtn += "{0}".format(mapping[col][row[col]])
        rtn += "\n"

    with open("data\mapping.json","w") as fl:
        json.dump(mapping, fl)
        
    with open("data\encoded.csv", "w") as fl:
        fl.write(rtn)



