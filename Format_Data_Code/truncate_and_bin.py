from datetime import datetime
from copy import deepcopy

scoreBins = { -22 :"-22" , -15 : "-15 | -21", -10: "-10 | -14", -7: "-7 | -9", -5: "-5 | -6", -3: "-3 | -4", -1: "-1 | -2", 
                 1 : "1 | 2", 3 : "3 | 4", 5 : "5 | 6", 7 : "7 | 9", 10 : "10 | 14", 15 : "15 | 21", 22 : "22+"}

# scoreBins = { -22 :"-7" , -15 : "-6", -10: "-5", -7: "-4", -5: "-3", -3: "-2", -1: "-1", 
#                 1 : "1", 3 : "2", 5 : "3", 7 : "4", 10 : "5", 15 : "6", 22 : "7"}

if __name__ == "__main__":
    rtnFile = ""
    headerLine = ""
    try:
        with open('data/spreadspoke_scores.csv') as fl:
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
                        rtn += ",".join(line).replace("\n", "") + ","
                        rtn += ",".join([str(cur_year_record[line[4]]["win"]), str(cur_year_record[line[4]]["loss"]), str(cur_year_record[line[4]]["tie"]), str(cur_year_record[line[7]]["win"]), str(cur_year_record[line[7]]["loss"]), str(cur_year_record[line[7]]["tie"])])
                        for i in scoreBins:
                            if scoreDiff > 0:
                                rtn += ",Home_Win\n"
                                break
                            elif scoreDiff < 0:
                                rtn += ",Home_Loss\n"
                                break
                            else:
                                rtn += ",Tie\n"
                                break
                            prevKey = i
                        rtnFile += rtn
                        if scoreDiff > 0:
                            cur_year_record[line[4]]["win"] += 1
                            cur_year_record[line[7]]["loss"] += 1
                        elif scoreDiff < 0:
                            cur_year_record[line[4]]["loss"] += 1
                            cur_year_record[line[7]]["win"] += 1
                        else:
                            cur_year_record[line[4]]["tie"] += 1
                            cur_year_record[line[7]]["tie"] += 1
                except Exception as e:
                    print("bad line " + str(e))
    except Exception as e:
        print(e)
    with open('data/cleaned_data.csv', 'w') as fl:
        fl.write((headerLine.replace('\n', "") + ",home_win,home_loss,home_tie,away_win,away_loss,away_tie,score_bin\n"))
        fl.write(rtnFile)



                

