from datetime import datetime
from copy import deepcopy

scoreBins = { -22 :"-22" , -15 : "-15 | -21", -10: "-10 | -14", -7: "-7 | -9", -5: "-5 | -6", -3: "-3 | -4", -1: "-1 | -2", 
                1 : "1 | 2", 3 : "3 | 4", 5 : "5 | 6", 7 : "7 | 9", 10 : "10 | 14", 15 : "15 | 21", 22 : "22+"}

if __name__ == "__main__":
    rtnFile = ""
    headerLine = ""
    try:
        with open('../data/spreadspoke_scores.csv') as fl:
            headerLine = fl.readline()
            lines = fl.readlines()
            teams = {}
            for line in lines:
                line = line.replace("\n", "").split(',')
                if line[4] not in teams:
                    teams[line[4]] = { "win" : 0, "loss": 0, "tie": 0}
                if line[7] not in teams:
                    teams[line[7]] = { "win" : 0, "loss": 0, "tie" : 0}
            cur_year_record = deepcopy(teams)
            prev_year = 2000
            for line in lines:
                try:
                    rtn = ""
                    line = line.replace("\n", "").split(',')
                    if int(line[1]) > 1999:
                        if prev_year != int(line[1]):
                            prev_year = int(line[1])
                            cur_year_record = deepcopy(teams)
                        scoreDiff = int(line[5]) - int(line[6])
                        prevKey = -100
                        for i in scoreBins:
                            if scoreDiff >= 22:
                                line.append(scoreBins[22])
                                rtn = ",".join(line).replace("\n", "")
                                break
                            elif scoreDiff <= -22:
                                line.append(scoreBins[-22])
                                rtn = ",".join(line).replace("\n", "")
                                break
                            elif prevKey <= scoreDiff <= i:
                                line.append(scoreBins[prevKey])
                                rtn = ",".join(line).replace("\n", "")
                                break
                            prevKey = i
                        rtnFile += "{0},({1} - {2} - {3}),({4} - {5} -{6})\n".format(rtn,cur_year_record[line[4]]["win"], cur_year_record[line[4]]["loss"], cur_year_record[line[4]]["tie"], cur_year_record[line[7]]["win"], cur_year_record[line[7]]["loss"], cur_year_record[line[7]]["tie"])

                        if scoreDiff > 0:
                            cur_year_record[line[4]]["win"] += 1
                            cur_year_record[line[7]]["loss"] += 1
                        elif scoreDiff < 0:
                            cur_year_record[line[4]]["loss"] += 1
                            cur_year_record[line[7]]["win"] += 1
                        else:
                            cur_year_record[line[4]]["tie"] += 1
                            cur_year_record[line[7]]["tie"] += 1
                except:
                    print("bad line")
    except Exception as e:
        print(e)
    with open('../data/cleaned_data.csv', 'w') as fl:
        fl.write((headerLine.replace('\n', "") + ",score_bin, home_team_record, away_team_record \n"))
        fl.write(rtnFile)



                

