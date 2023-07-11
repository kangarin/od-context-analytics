# read a file, every line is a number, draw a line chart
import matplotlib.pyplot as plt
import numpy as np

def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [float(line) for line in lines]

def draw_line_chart(data, title, xlabel, ylabel):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == '__main__':
    data = read_file('car_fps_profile/fps_sw1.txt')
    draw_line_chart(data, 'proposed fps with sliding window of size 1', '5 * frame', 'fps')