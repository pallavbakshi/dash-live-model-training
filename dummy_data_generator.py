import csv
import random

import os

if os.path.exists("eggs.csv"):
    os.remove("eggs.csv")

for x in range(1000000):
    with open('eggs.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        print(x)

        spamwriter.writerow([x, random.random()])