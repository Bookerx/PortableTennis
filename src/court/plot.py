import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        pass

    def visualize(self):
        pass


def cal_dis(coor1, coor2):
    out = np.square(coor1[0] - coor2[0]) + np.square(coor1[1] - coor2[1])
    return np.sqrt(out)


def calculate_movement(ls):
    move = []
    for idx in range(len(ls)-1):
        move.append(cal_dis(ls[idx], ls[idx+1]))
    return move

def plot_speed(player1, player2, max_time=20):
    if len(player1) < 2 or len(player2) < 2:
        array1 = np.array([[0,0], [0,0]])
        array2 = np.array([[0,0], [0,0]])
        # cv2.imwrite("speed_tmp.png", np.zeros((100, 100)))
        # return
    else:
        array1 = calculate_movement(player1)
        array2 = calculate_movement(player2)
    # Example arrays (replace with your own data)

    # Determine the number of elements to plot
    num_elements = min(max_time, len(array1), len(array2))

    # Extract the last 20 elements (or fewer if the array is shorter)
    valid_array1 = array1[-num_elements:]
    valid_array2 = array2[-num_elements:]

    # Plot the curves
    plt.plot(valid_array1, label='Player 1')
    plt.plot(valid_array2, label='Player 2')

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.title('Speed of players')

    # Add legend
    plt.legend()

    # Display the plot
    plt.savefig("tmp/speed_tmp.png")
    plt.clf()


def calculate_point_frequencies(area_width, area_height, points, grid_rows, grid_cols):
    grid_width = area_width // grid_cols
    grid_height = area_height // grid_rows
    frequencies = [[0] * grid_cols for _ in range(grid_rows)]

    for point in points:
        x, y = point
        grid_x = int(x // grid_width)
        grid_y = int(y // grid_height)
        frequencies[grid_y][grid_x] += 1

    return frequencies


if __name__ == '__main__':
    area_width = 12
    area_height = 16
    points = [(7, 1), (4, 1)]
    grid_rows = 2
    grid_cols = 4

    frequencies = calculate_point_frequencies(area_width, area_height, points, grid_rows, grid_cols)
    for row in frequencies:
        print(row)
