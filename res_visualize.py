# read a file, every line is a number, draw a line chart
import matplotlib.pyplot as plt
import numpy as np

def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [float(line) for line in lines]

def draw_line_chart(data, title, xlabel, ylabel):
    # set font size to large
    plt.rcParams.update({'font.size': 36})
    plt.figure(figsize=(15, 10))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()
    plt.savefig('res.png')

if __name__ == '__main__':
    data = read_file('res.txt')
    draw_line_chart(data, 'proposed resolution change over time', 'frame ID', 'resolution')