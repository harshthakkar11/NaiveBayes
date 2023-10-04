# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
from math import sqrt
from math import exp
from math import pi


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


def readFile(dataset_path):
    input_data = fetch_data(dataset_path)
    # input_np = np.array(input_data)
    input_np = list(input_data)
    return input_np


# load data and cleaning it
training_data = r'1a-training.txt'
test_data = r'1a-test.txt'
large_120_data = r'1c-data.txt'

train_np = readFile(training_data)
print(train_np)
test_np = readFile(test_data)
print(test_np)
large_np = readFile(large_120_data)
checkr = readFile(large_120_data)


def predict(test_np, train_np):
    print("prediction for test data :")
    # converting data to float
    for i in range(len(train_np)):
        for j in range(len(train_np[i]) - 1):
            train_np[i][j] = float(train_np[i][j])
    for i in range(len(test_np)):
        for j in range(len(test_np[i])):
            test_np[i][j] = float(test_np[i][j])
    tmheight = 0
    tmweight = 0
    tmage = 0
    twheight = 0
    twweight = 0
    twage = 0
    mcount = 0
    wcount = 0
    # finding mean of data based on label
    for i in range(len(train_np)):
        if train_np[i][3] == 'M':
            tmheight = train_np[i][0] + tmheight
            tmweight = train_np[i][1] + tmweight
            tmage = train_np[i][2] + tmage
            mcount = mcount + 1
        else:
            twheight = train_np[i][0] + twheight
            twweight = train_np[i][1] + twweight
            twage = train_np[i][2] + twage
            wcount = wcount + 1
    hmmean = tmheight / mcount
    wmmean = tmweight / mcount
    ammean = tmage / mcount
    hwmean = twheight / wcount
    wwmean = twweight / wcount
    awmean = twage / wcount
    smheight = 0
    smweight = 0
    smage = 0
    swheight = 0
    swweight = 0
    swage = 0

    # finding standard deviation of each column and for different label
    for i in range(len(train_np)):
        if train_np[i][3] == 'M':
            smheight = ((train_np[i][0] - hmmean) ** 2) + smheight
            smweight = (train_np[i][1] - wmmean) ** 2 + smweight
            smage = (train_np[i][2] - ammean) ** 2 + smage
        else:
            swheight = (train_np[i][0] - hwmean) ** 2 + swheight
            swweight = (train_np[i][1] - wwmean) ** 2 + swweight
            swage = (train_np[i][2] - awmean) ** 2 + swage
    stdmh = sqrt((smheight / (mcount - 1)))
    stdmw = sqrt((smweight / (mcount - 1)))
    stdma = ((smage) ** 1 / 2) / (mcount - 1)
    stdwh = ((swheight) ** 1 / 2) / (wcount - 1)
    stdww = ((swweight) ** 1 / 2) / (wcount - 1)
    stdwa = ((swage) ** 1 / 2) / (wcount - 1)
    pm = mcount / len(train_np)
    pw = wcount / len(train_np)
    # finding probability of each column given label
    for i in range(len(test_np)):
        xMinusMeu = (test_np[i][0] - hmmean)
        xMinusMeuSquare = xMinusMeu ** 2
        sigma2Sq = 2 * (stdmh ** 2)
        temp = (-xMinusMeuSquare * 1.0) / sigma2Sq
        denominator = sqrt(2 * pi) * stdmh
        phgm = (math.exp(temp) * 1.0) / denominator
        temp2 = (-((test_np[i][1] - wmmean) ** 2) / (2 * (stdmw ** 2)))
        exponent2 = math.exp(temp2)
        pwgm = ((1 / ((sqrt(2 * pi)) * stdmw)) * exponent2)
        temp3 = (-((test_np[i][2] - ammean) ** 2) / (2 * (stdma ** 2)))
        exponent3 = math.exp(temp3)
        pagm = ((1 / ((sqrt(2 * pi)) * stdmw)) * exponent3)
        temp4 = (-(((test_np[i][0] - hwmean) ** 2) / ((2 * (stdwh ** 2)))))
        exponent4 = math.exp(temp4)
        phgw = (1 / ((sqrt(2 * pi) * stdwh))) * exponent4
        temp5 = (-(((test_np[i][1] - wwmean) ** 2) / ((2 * (stdww ** 2)))))
        exponent5 = math.exp(temp5)
        pwgw = (1 / ((sqrt(2 * pi) * stdwh))) * exponent5
        temp6 = (-(((test_np[i][1] - awmean) ** 2) / ((2 * (stdwa ** 2)))))
        exponent6 = math.exp(temp6)
        pagw = (1 / ((sqrt(2 * pi) * stdwa))) * exponent6
        pom1 = pm * phgm * pwgm * pagm
        pow1 = pw * phgw * pwgw * pagw
        pom = pom1 / (pom1 + pow1)
        pow = pow1 / (pom1 + pow1)
        # predicting gender
        if pom > pow:
            print(test_np[i], "M")
        else:
            print(test_np[i], "W")


predict(test_np, train_np)


# predicting gender for each program data using height, weight and age

def predict1(large_np):
    # converting data to float
    for i in range(len(large_np)):
        for j in range(len(large_np[i]) - 1):
            large_np[i][j] = float(large_np[i][j])
    tmheight = 0
    tmweight = 0
    tmage = 0
    twheight = 0
    twweight = 0
    twage = 0
    mcount = 0
    wcount = 0
    # finding standard deviation of each column and for different label
    for i in range(len(large_np)):
        if large_np[i][3] == 'M':
            tmheight = large_np[i][0] + tmheight
            tmweight = large_np[i][1] + tmweight
            tmage = large_np[i][2] + tmage
            mcount = mcount + 1
        else:
            twheight = large_np[i][0] + twheight
            twweight = large_np[i][1] + twweight
            twage = large_np[i][2] + twage
            wcount = wcount + 1
    hmmean = tmheight / mcount
    wmmean = tmweight / mcount
    ammean = tmage / mcount
    hwmean = twheight / wcount
    wwmean = twweight / wcount
    awmean = twage / wcount
    smheight = 0
    smweight = 0
    smage = 0
    swheight = 0
    swweight = 0
    swage = 0
    pm = mcount / len(large_np)
    pw = wcount / len(large_np)
    # finding probability of each column given label
    for i in range(len(large_np)):
        for j in range(len(large_np)):
            if i != j:
                if large_np[j][3] == 'M':
                    smheight = ((large_np[j][0] - hmmean) ** 2) + smheight
                    smweight = (large_np[j][1] - wmmean) ** 2 + smweight
                    smage = (large_np[j][2] - ammean) ** 2 + smage
                else:
                    swheight = (large_np[j][0] - hwmean) ** 2 + swheight
                    swweight = (large_np[j][1] - wwmean) ** 2 + swweight
                    swage = (large_np[j][2] - awmean) ** 2 + swage
            stdmh = sqrt((smheight / (mcount - 1)))
            stdmw = sqrt((smweight / (mcount - 1)))
            stdma = sqrt((smage) / (mcount - 1))
            stdwh = sqrt(((swheight) ** 1 / 2) / (wcount - 1))
            stdww = sqrt(((swweight) ** 1 / 2) / (wcount - 1))
            stdwa = sqrt(((swage) ** 1 / 2) / (wcount - 1))

        xMinusMeu = (large_np[i][0] - hmmean)
        xMinusMeuSquare = xMinusMeu ** 2
        sigma2Sq = 2 * (stdmh ** 2)
        temp = (-xMinusMeuSquare * 1.0) / sigma2Sq
        denominator = sqrt(2 * pi) * stdmh
        phgm = (math.exp(temp) * 1.0) / denominator
        temp2 = (-((large_np[i][1] - wmmean) ** 2) / (2 * (stdmw ** 2)))
        exponent2 = math.exp(temp2)
        pwgm = ((1 / ((sqrt(2 * pi)) * stdmw)) * exponent2)
        temp3 = (-((large_np[i][2] - ammean) ** 2) / (2 * (stdma ** 2)))
        exponent3 = math.exp(temp3)
        pagm = ((1 / ((sqrt(2 * pi)) * stdma)) * exponent3)
        temp4 = (-(((large_np[i][0] - hwmean) ** 2) / ((2 * (stdwh ** 2)))))
        exponent4 = math.exp(temp4)
        phgw = ((1 / ((sqrt(2 * pi) * stdwh))) * exponent4)
        temp5 = (-(((large_np[i][1] - wwmean) ** 2) / ((2 * (stdww ** 2)))))
        exponent5 = math.exp(temp5)
        pwgw = ((1 / ((sqrt(2 * pi) * stdwh))) * exponent5)
        temp6 = (-(((large_np[i][1] - awmean) ** 2) / ((2 * (stdwa ** 2)))))
        exponent6 = math.exp(temp6)
        pagw = ((1 / ((sqrt(2 * pi) * stdwa))) * exponent6)
        pom1 = pm * phgm * pwgm * pagm
        pow1 = pw * phgw * pwgw * pagw
        pom = pom1 / (pom1 + pow1)
        pow = pow1 / (pom1 + pow1)
        if pom > pow:
            large_np[i][3] = 'M'
        else:
            large_np[i][3] = 'W'
    per = 0

    # checking accuracy
    for i in range(len(large_np)):
        if checkr[i][3] == large_np[i][3]:
            per = per + 1
    pert = per / len(large_np) * 100
    print("Accuracy using height,weight and age in Gaussian Na ̈ıve Bayes ", pert)


predict1(large_np)


# predicting gender for each program data using height and weight only
def predict2(large_np):
    for i in range(len(large_np)):
        for j in range(len(large_np[i]) - 1):
            large_np[i][j] = float(large_np[i][j])
    tmheight = 0
    tmweight = 0
    twheight = 0
    twweight = 0
    mcount = 0
    wcount = 0
    for i in range(len(large_np)):
        if large_np[i][3] == 'M':
            tmheight = large_np[i][0] + tmheight
            tmweight = large_np[i][1] + tmweight
            mcount = mcount + 1
        else:
            twheight = large_np[i][0] + twheight
            twweight = large_np[i][1] + twweight
            wcount = wcount + 1
    hmmean = tmheight / mcount
    wmmean = tmweight / mcount
    hwmean = twheight / wcount
    wwmean = twweight / wcount
    smheight = 0
    smweight = 0
    swheight = 0
    swweight = 0
    pm = mcount / len(large_np)
    pw = wcount / len(large_np)
    for i in range(len(large_np)):
        for j in range(len(large_np)):
            if i != j:
                if large_np[j][3] == 'M':
                    smheight = ((large_np[j][0] - hmmean) ** 2) + smheight
                    smweight = (large_np[j][1] - wmmean) ** 2 + smweight
                else:
                    swheight = (large_np[j][0] - hwmean) ** 2 + swheight
                    swweight = (large_np[j][1] - wwmean) ** 2 + swweight
            stdmh = sqrt((smheight / (mcount - 1)))
            stdmw = sqrt((smweight / (mcount - 1)))
            stdwh = sqrt(((swheight) ** 1 / 2) / (wcount - 1))
            stdww = sqrt(((swweight) ** 1 / 2) / (wcount - 1))

        xMinusMeu = (large_np[i][0] - hmmean)
        xMinusMeuSquare = xMinusMeu ** 2
        sigma2Sq = 2 * (stdmh ** 2)
        temp = (-xMinusMeuSquare * 1.0) / sigma2Sq
        denominator = sqrt(2 * pi) * stdmh
        phgm = (math.exp(temp) * 1.0) / denominator
        # phgm = (1/((2*(3.14))*stdmh))*(math.exp(1/((test_np[i][0]-hmmean)/(2*(stdmh**2)))))
        # print("=====================")
        temp2 = (-((large_np[i][1] - wmmean) ** 2) / (2 * (stdmw ** 2)))
        exponent2 = math.exp(temp2)
        pwgm = ((1 / ((sqrt(2 * pi)) * stdmw)) * exponent2)
        temp4 = (-(((large_np[i][0] - hwmean) ** 2) / ((2 * (stdwh ** 2)))))
        exponent4 = math.exp(temp4)
        phgw = ((1 / ((sqrt(2 * pi) * stdwh))) * exponent4)
        temp5 = (-(((large_np[i][1] - wwmean) ** 2) / ((2 * (stdww ** 2)))))
        exponent5 = math.exp(temp5)
        pwgw = ((1 / ((sqrt(2 * pi) * stdwh))) * exponent5)
        pom1 = pm * phgm * pwgm
        pow1 = pw * phgw * pwgw
        pom = pom1 / (pom1 + pow1)
        pow = pow1 / (pom1 + pow1)
        if pom > pow:
            large_np[i][3] = 'M'
        else:
            large_np[i][3] = 'W'
    per = 0
    for i in range(len(large_np)):
        if checkr[i][3] == large_np[i][3]:
            per = per + 1
    pert = per / len(large_np) * 100
    print("Accuracy using only height and weight in Gaussian Na ̈ıve Bayes ", pert)


predict2(large_np)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
