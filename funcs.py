import time
import numpy as np
import configuration as config
from tqdm import trange
import matplotlib.pyplot as plt

func = lambda x, y, center, sigma: np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)) / (
        2 * np.pi * sigma ** 2)


# ЧБ
def black_and_white(arr):
    X, Y, U = arr.shape
    result = np.zeros((X, Y), dtype=int)
    for i in range(X):
        for j in range(Y):
            result[i, j] = np.mean(arr[i, j])
    return np.array(result, dtype=np.uint8)


# Формирование фильтра Гаусса определённой размерности
def gauss(shape):
    matrix = np.zeros((shape, shape))
    center = shape // 2
    for i in range(shape):
        for j in range(shape):
            matrix[i, j] = func(i, j, center, config.sigma)
    matrix /= np.sum(matrix)
    return matrix


# Наложение фильтра Гаусса
def filter(arr, shape):
    X, Y = arr.shape
    center = shape // 2
    borderX, borderY = arr.shape
    # Добавляем рамку
    borderX += center * 2
    borderY += center * 2
    result = np.zeros((borderX, borderY), dtype=np.uint8)
    # Внутрь помещаем изображение
    result[center:-center, center:-center] = arr
    # Генерим фильтр
    matrix = gauss(shape)
    # Накладываем фильтр
    for i in range(X):
        for j in range(Y):
            result[i + center, j + center] = int(np.sum(result[i:i + shape, j:j + shape] * matrix))
    return result[center:-center, center:-center]


# Получаем точки для формирования круга Брезенхема
def brezenham_circle(arr, x, y):
    return [arr[x - 3, y], arr[x - 3, y + 1], arr[x - 2, y + 2], arr[x - 1, y + 3], arr[x, y + 3],
            arr[x + 1, y + 3], arr[x + 2, y + 2], arr[x + 3, y + 1], arr[x + 3, y],
            arr[x + 3, y - 1], arr[x + 2, y - 2], arr[x + 1, y - 3], arr[x, y - 3],
            arr[x - 1, y - 3], arr[x - 2, y - 2], arr[x - 3, y - 1]]


# Детектирование особых точек
def FAST(arr):
    keys_points = np.zeros(shape=arr.shape)
    time.sleep(0.1)
    # проход по всем пикселям кроме крайних
    for i in trange(3, arr.shape[0] - 3):
        for j in range(3, arr.shape[1] - 3):
            circle = brezenham_circle(arr, i, j)
            p = arr[i, j]
            # проверяем 1, 9
            if ((circle[0] > p + config.t) and (circle[8] < p - config.t)) or (
                    (circle[0] < p - config.t) and (circle[8] > p + config.t)):
                continue
            # проверяем 5, 13
            elif ((circle[4] > p + config.t) and (circle[12] < p - config.t)) or (
                    (circle[4] < p - config.t) and (circle[12] > p + config.t)):
                continue
            for pix in range(16):  # каждая точка из окружности
                if np.size(np.argwhere(np.array(
                        [circle[i - pix] for i in range(np.size(circle))][:12]) > p + config.t)) == 12 or np.size(
                    np.argwhere(
                        np.array([circle[i - pix] for i in range(np.size(circle))][:12]) < p - config.t)) == 12:
                    keys_points[i, j] = 255
                    break
    return keys_points


# Отклик по Харрису
def Harris(arr, keys_points):
    R_list = []
    coordsArr = np.argwhere(keys_points == 255)
    Gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # Элементы весового окна
    gaussian = gauss(5)
    time.sleep(0.1)
    # Цикл по всем точкам из FAST
    for point in trange(coordsArr.shape[0]):
        M = np.zeros(shape=(2, 2))
        for i in range(-2, 3):
            for j in range(-2, 3):
                x, y = coordsArr[point][0] + i, coordsArr[point][1] + j
                # Частные производные
                neighbours = arr[x - 1: x + 2, y - 1: y + 2]
                Ix = np.sum(Gx * neighbours)
                Iy = np.sum(Gy * neighbours)
                # Матрица А
                A = np.array([[Ix ** 2, Ix * Iy], [Ix * Iy, Iy ** 2]])
                # Структурный тензор
                M = M + gaussian[i + 2, j + 2] * A
        # Собств. числа матрицы M
        lambda1, lambda2 = np.linalg.eigvals(M)[0], np.linalg.eigvals(M)[1]
        det, trace = lambda1 * lambda2, lambda1 + lambda2
        # Функция отклика
        R = det - config.k * (trace ** 2)
        # Отбираем выше некоторого порога
        if R > config.threshold:
            R_list.append([coordsArr[point][0], coordsArr[point][1]])
    return np.array(R_list)


# Вычисление ориентации
def orientation(R_arr, arr):
    orient_list = []
    time.sleep(0.1)
    for point in trange(R_arr.shape[0]):
        # Область вокруг особой точки
        field = [[0, 0]]
        # Рамка для области
        borders = np.zeros(shape=(arr.shape[0] + 2 * 31, arr.shape[1] + 2 * 31))
        borders[31:-31, 31:-31] = arr
        # Заполняем область вокруг особой точки
        for r in range(1, 32):
            for x in range(-r, r + 1):
                # Проверяем отрицательные квадраты y
                if r ** 2 - x ** 2 < 0:
                    continue
                y1 = int(np.sqrt(r ** 2 - x ** 2))
                y2 = int(-np.sqrt(r ** 2 - x ** 2))
                if -r <= y1 <= r:
                    field.append([x, y1])
                if -r <= y2 <= r:
                    field.append([x, y2])
            for y in range(-r, r + 1):
                # То же самое с квадратами x
                if r ** 2 - y ** 2 < 0:
                    continue
                x1 = int(np.sqrt(r ** 2 - y ** 2))
                x2 = int(-np.sqrt(r ** 2 - y ** 2))
                if -r <= x1 <= r:
                    field.append([x1, y])
                if -r <= x2 <= r:
                    field.append([x2, y])
        # Исключаем повторяющиеся
        field = np.unique(field, axis=0)
        # вычисляем моменты
        m00, m01, m10 = 0, 0, 0
        for p in field:
            I = borders[p[0] + R_arr[point][0] + 31, p[1] + R_arr[point][1] + 31]
            m00 += I
            m01 += p[1] * I
            m10 += p[0] * I
        teta = np.arctan2(m01, m10)
        if teta < 0:
            orient_list.append(np.arctan2(m01, m10) + np.pi * 2)
        else:
            orient_list.append(np.arctan2(m01, m10))
    return np.array(orient_list)


def rotate_mtrx(teta):
    return np.array([[np.cos(teta), np.sin(teta)], [-np.sin(teta), np.cos(teta)]])


# Построение дескриптора
def BRIEF(arr, R_arr, orient_arr, n):
    descriptors = []
    # Добавляем рамку
    borders = np.zeros(shape=(arr.shape[0] + 30, arr.shape[1] + 30))
    borders[15:-15, 15:-15] = arr
    pattern_points = []
    time.sleep(0.1)
    # Цикл по особым точкам
    for k in trange(R_arr.shape[0]):
        x_c, y_c = R_arr[k][0] + 15, R_arr[k][1] + 15
        teta_c = orient_arr[k]
        # Область вокруг особой точки
        field = [[0, 0]]
        # Рассматриваем радиус
        for r in range(1, 16):
            for x in range(x_c - r, x_c + r + 1):
                if r ** 2 - (x - x_c) ** 2 < 0:
                    continue
                y1 = int(np.round(np.sqrt(r ** 2 - (x - x_c) ** 2) + y_c))
                y2 = int(np.round(-np.sqrt(r ** 2 - (x - x_c) ** 2) + y_c))
                if 0 <= y1 < borders.shape[1]:
                    field.append([x - x_c, y1 - y_c])
                if 0 <= y2 < borders.shape[1]:
                    field.append([x - x_c, y2 - y_c])
            for y in range(y_c - r, y_c + r + 1):
                if r ** 2 - (y - y_c) ** 2 < 0:
                    continue
                x1 = int(np.round(np.sqrt(r ** 2 - (y - y_c) ** 2) + x_c))
                x2 = int(np.round(-np.sqrt(r ** 2 - (y - y_c) ** 2) + x_c))
                if 0 <= x1 < borders.shape[0]:
                    field.append([x1 - x_c, y - y_c])
                if 0 <= x2 < borders.shape[0]:
                    field.append([x2 - x_c, y - y_c])
        field = np.unique(field, axis=0)
        # Учёт ориентации
        S = np.zeros(shape=(2, n, 2))
        for i in range(n):
            if k == 0:
                while True:
                    # Чтобы пары точек не повторялись
                    rand = np.random.normal(0, 31 ** 2 / 25, 4).astype(int)
                    if [rand[0], rand[1]] in field.tolist() and [rand[2], rand[3]] in field.tolist():
                        if not [rand[0], rand[1]] in pattern_points and not [rand[2], rand[3]] in pattern_points:
                            pattern_points.append([rand[0], rand[1]])
                            pattern_points.append([rand[2], rand[3]])
                            S[0, i] = [rand[0], rand[1]]
                            S[1, i] = [rand[2], rand[3]]
                            break
            else:
                u1, u2 = pattern_points[2 * i], pattern_points[2 * i + 1]
                S[0, i] = [u1[0], u1[1]]
                S[1, i] = [u2[0], u2[1]]
        # создаем набор углов и округляем ориентацию особой точки
        angles = np.zeros(30)
        for i in range(len(angles)):
            if i == 0:
                angles[i] = 0
            else:
                angles[i] = angles[i - 1] + 2 * np.pi / 30
        teta_c = min(angles, key=lambda x: abs(x - teta_c))
        # Подсчет S_teta
        _S1, _S2 = S[0].T, S[1].T
        S1 = (rotate_mtrx(teta_c) @ _S1).astype(int)
        S2 = (rotate_mtrx(teta_c) @ _S2).astype(int)
        # Вычисление дескриптора
        bin_row = np.zeros(shape=n)
        for i in range(n):
            p1, p2 = S1.T[i], S2.T[i]
            if borders[x_c + p1[0], y_c + p1[1]] < borders[x_c + p2[0], y_c + p2[1]]:
                bin_row[i] = 1
        descriptors.append(list(bin_row))
    return descriptors


def to_file(descriptors):
    with open('descriptors.txt', 'w') as file:
        for d in descriptors:
            file.write(str(d) + '\n')


def drawing(R_arr, arr):
    for point in range(R_arr.shape[0]):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (0 > R_arr[point][0] + i or R_arr[point][0] + i >= arr.shape[0]) or (
                        0 > R_arr[point][1] + j or R_arr[point][1] + j >= arr.shape[1]):
                    continue
                arr[R_arr[point][0] + i, R_arr[point][1] + j] = [255, 0, 0]


def apply(image):
    bw_image = black_and_white(image)
    print("Детектирование особых точек")
    key_points = FAST(bw_image)
    print(f"До фильтрации: {np.size(np.argwhere(key_points == 255), axis=0)} кандидатов")
    key_points = Harris(bw_image, key_points)
    print(f"После фильтрации: {len(key_points)} кандидатов")
    print("Находим ориентации")
    orient_arr = orientation(key_points, bw_image)
    drawing(key_points, image)
    plt.imshow(image)
    plt.show()
    print("Построение дескрипторов")
    descriptors = BRIEF(filter(bw_image, 5), key_points, orient_arr, 256)
    print("Запись на диск")
    to_file(descriptors)
