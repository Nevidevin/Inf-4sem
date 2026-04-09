import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from PIL.ImageColor import colormap
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import interp1d, splprep, splev
L=10.5/100
g=9.81
A1=(15+36/3)/1000
B1=(15+36/2)/1000
h0=0.005
mu0 = 4 * np.pi * 1e-7
z0 = 1.5 / 100
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import interp1d


def animate_trajectories_without_tail(trajectories_data, magnets, filename='trajectories.gif',
                                      duration=10, fps=30, dpi=100, magnet_size=20):
    """
    Создаёт анимацию движения нескольких траекторий (след остаётся навсегда) и отображает магниты.

    Параметры:
    ----------
    trajectories_data : list of lists
        Каждый элемент: [T, X, Y], где T, X, Y — массивы времени и координат.
        T должен начинаться с 0 (или будет сдвинут).
    magnets : list of lists
        Каждый элемент: [x, y, strength], где strength > 0 - зелёный, strength < 0 - красный
    filename : str
        Имя выходного файла (поддерживаются .gif, .mp4).
    duration : float
        Длительность видео в секундах.
    fps : int
        Кадров в секунду.
    dpi : int
        Разрешение для сохранения.
    magnet_size : float
        Размер маркера магнита (s для scatter).
    """
    n_frames = int(duration * fps)
    time_grid = np.linspace(0, duration, n_frames)

    # Интерполяция всех траекторий на общую временную сетку
    interpolated = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories_data)))  # разные цвета

    # Определяем общие границы для осей
    all_x = []
    all_y = []

    for T, X, Y in trajectories_data:
        T = np.asarray(T)
        X = np.asarray(X)
        Y = np.asarray(Y)
        # Приводим время к началу с 0
        if T[0] != 0:
            T = T - T[0]
        # Интерполяторы
        interp_x = interp1d(T, X, kind='linear', fill_value=(X[0], X[-1]), bounds_error=False)
        interp_y = interp1d(T, Y, kind='linear', fill_value=(Y[0], Y[-1]), bounds_error=False)
        # Значения на временной сетке (обрезка по длительности)
        X_grid = interp_x(time_grid)
        Y_grid = interp_y(time_grid)
        interpolated.append((X_grid, Y_grid))
        all_x.extend(X_grid)
        all_y.extend(Y_grid)

    # Добавляем координаты магнитов для границ
    for mx, my, _ in magnets:
        all_x.append(mx)
        all_y.append(my)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin = 0.05 * max(x_max - x_min, y_max - y_min)
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)

    # Настройка рисунка
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('X,м')
    ax.set_ylabel('Y,м')
    ax.set_title('Движение маятника')

    # Хранилища для линий-следов и точек
    lines = []
    points = []
    for i, (X_grid, Y_grid) in enumerate(interpolated):
        line, = ax.plot([], [], lw=2, color=colors[i], alpha=0.7, label=f'Траектория {i + 1}')
        point, = ax.plot([], [], 'o', color=colors[i], markersize=6, zorder=5)
        lines.append(line)
        points.append(point)

    # Рисуем магниты
    magnet_x = [m[0] for m in magnets]
    magnet_y = [m[1] for m in magnets]
    magnet_colors = ['red' if m[2] > 0 else 'green' for m in magnets]

    magnet_scatter = ax.scatter(magnet_x, magnet_y, s=magnet_size,
                                c=magnet_colors, marker='o',)


    # Легенда
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=8, label='Магнит - притяжение'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Магнит - отталкивание')
    ]

    for i in range(len(colors)):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=colors[i], markersize=8,
                                          label=f'Траектория {i + 1}'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def update(frame):
        for i, (X_grid, Y_grid) in enumerate(interpolated):
            # След от начала до текущего кадра
            lines[i].set_data(X_grid[:frame + 1], Y_grid[:frame + 1])
            # Точка в текущем положении
            points[i].set_data([X_grid[frame]], [Y_grid[frame]])
        # Возвращаем все объекты для blit (опционально)
        return lines + points + [magnet_scatter]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    # Сохранение в файл
    if filename.endswith('.gif'):
        writer = PillowWriter(fps=fps)
    else:
        # Для mp4 нужен ffmpeg
        writer = 'ffmpeg'
    ani.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Анимация сохранена в {filename}")

    return ani
def animate_trajectories(trajectories_data, magnets, filename='trajectories_with_magnets.gif',
                                      duration=10, fps=30, trail_length=100, dpi=100,
                                      line_width=2, fade_alpha=True, magnet_size=20):
    """
    Создаёт анимацию движения траекторий с исчезающим хвостом-линией и магнитами.

    Параметры:
    ----------
    trajectories_data : list of lists
        Каждый элемент: [T, X, Y] - время и координаты траектории
    magnets : list of lists
        Каждый элемент: [x, y, strength] - координаты и сила магнита
        strength > 0 - зелёный, strength < 0 - красный
    filename : str
        Имя выходного файла
    duration : float
        Длительность видео в секундах
    fps : int
        Кадров в секунду
    trail_length : int
        Длина хвоста в кадрах
    dpi : int
        Разрешение
    line_width : float
        Толщина линии хвоста
    fade_alpha : bool
        Плавное исчезновение хвоста
    magnet_size : float
        Размер маркера магнита (s для scatter)
    """
    n_frames = int(duration * fps)
    time_grid = np.linspace(0, duration, n_frames)

    # Интерполяция всех траекторий
    interpolated = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories_data)))

    all_x, all_y = [], []

    for T, X, Y in trajectories_data:
        T = np.asarray(T)
        X = np.asarray(X)
        Y = np.asarray(Y)
        if T[0] != 0:
            T = T - T[0]
        interp_x = interp1d(T, X, kind='linear', fill_value=(X[0], X[-1]), bounds_error=False)
        interp_y = interp1d(T, Y, kind='linear', fill_value=(Y[0], Y[-1]), bounds_error=False)
        X_grid = interp_x(time_grid)
        Y_grid = interp_y(time_grid)
        interpolated.append((X_grid, Y_grid))
        all_x.extend(X_grid)
        all_y.extend(Y_grid)

    # Добавляем координаты магнитов для границ
    for mx, my, _ in magnets:
        all_x.append(mx)
        all_y.append(my)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin = 0.05 * max(x_max - x_min, y_max - y_min)
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)

    # Настройка рисунка
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('X,м')
    ax.set_ylabel('Y,м')
    ax.set_title('Движение маятника.')

    # Рисуем магниты (статические, но перерисовываются в каждом кадре для совместимости с blit)
    magnet_scatter = None

    # Хранилища для линий хвоста и текущих точек
    trail_lines = []
    points = []

    for i, color in enumerate(colors):
        # Линия для хвоста
        line, = ax.plot([], [], lw=line_width, color=color, alpha=0.8, zorder=2)
        trail_lines.append(line)
        # Текущая позиция
        point, = ax.plot([], [], 'o', color=color, markersize=8, zorder=5)
        points.append(point)

        # Хранилище истории позиций для каждой траектории
        trail_lines[i].history = deque(maxlen=trail_length)

    # Подготовка данных для магнитов
    magnet_x = [m[0] for m in magnets]
    magnet_y = [m[1] for m in magnets]
    magnet_colors = ['red' if m[2] > 0 else 'green' for m in magnets]

    # Создаём scatter для магнитов (один раз)
    magnet_scatter = ax.scatter(magnet_x, magnet_y, s=magnet_size,
                                c=magnet_colors, marker='o')



    # Легенда
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=8, label='Магнит - притяжение'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Магнит - отталкивание')
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def update(frame):
        # Обновляем траектории
        for i, (X_grid, Y_grid) in enumerate(interpolated):
            if frame < len(X_grid):
                x_curr = X_grid[frame]
                y_curr = Y_grid[frame]

                # Добавляем текущую позицию в историю
                trail_lines[i].history.append((x_curr, y_curr))

                # Обновляем текущую точку
                points[i].set_data([x_curr], [y_curr])

                # Получаем все точки хвоста
                history_list = list(trail_lines[i].history)
                if len(history_list) >= 2:
                    xs = [p[0] for p in history_list]
                    ys = [p[1] for p in history_list]

                    # Опционально: сглаживание хвоста
                    if len(history_list) > 10:
                        try:
                            tck, u = splprep([xs, ys], s=0.01 * len(history_list), per=False)
                            u_new = np.linspace(0, 1, min(200, 5 * len(history_list)))
                            xs_smooth, ys_smooth = splev(u_new, tck)
                            trail_lines[i].set_data(xs_smooth, ys_smooth)
                        except:
                            trail_lines[i].set_data(xs, ys)
                    else:
                        trail_lines[i].set_data(xs, ys)

                    # Плавное исчезновение
                    if fade_alpha and len(history_list) > 1:
                        alphas = np.linspace(0.15, 0.9, len(history_list))
                        avg_alpha = np.mean(alphas)
                        trail_lines[i].set_alpha(avg_alpha)
                else:
                    trail_lines[i].set_data([], [])

        # Магниты не нужно обновлять, они статические
        # Но возвращаем их для blit (если используем blit=True)
        return points + trail_lines + [magnet_scatter]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    # Сохранение
    if filename.endswith('.gif'):
        writer = PillowWriter(fps=fps)
    else:
        writer = 'ffmpeg'
    ani.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Анимация с магнитами сохранена в {filename}")

    return ani
def aprocsimate(x,y):
    N = len(x)
    S1 = S2 = S3 = S5 = S6 = 0
    X = float(sum(x) / len(x))
    Y = float(sum(y) / len(y))
    for i in range(N):
        S1 += float((x[i] - X) ** 2)
        S2 += float((y[i] - Y) ** 2)
        S3 += float((x[i] - X) * (y[i] - Y))
        S5 += round(x[i] - X, 3)
        S6 += round(y[i] - Y, 3)
    print('k =', S3 / S1)
    k = S3 / S1
    print('r=', (S3 / S1) * ((S1 / (N - 1)) ** 0.5 / (S2 / (N - 1)) ** 0.5))
    r = float((S3 / S1) * ((S1 / (N - 1)) ** 0.5 / (S2 / (N - 1)) ** 0.5))
    print('B=', Y - (S3 / S1) * X)
    l = ((S2 / S1) - k ** 2) ** 0.5
    dk = (l / ((N - 2) ** 0.5))
    B = Y - (S3 / S1) * X
    print('dk=', (l / ((N - 2) ** 0.5)))
    print('dB=', dk * (S1 / N) ** 0.5)
    dB=dk * (S1 / N) ** 0.5
    return k,B,r
def cor(x,y):
    N = len(x)
    S1 = S2 = S3 = S5 = S6 = 0
    X = float(sum(x) / len(x))
    Y = float(sum(y) / len(y))
    for i in range(N):
        S1 += float((x[i] - X) ** 2)
        S2 += float((y[i] - Y) ** 2)
        S3 += float((x[i] - X) * (y[i] - Y))
    print('r=', (S3 / S1) * ((S1 / (N - 1)) ** 0.5 / (S2 / (N - 1)) ** 0.5))
    r = float((S3 / S1) * ((S1 / (N - 1)) ** 0.5 / (S2 / (N - 1)) ** 0.5))
    return r
for fsdf in range(1):#DATA
    Xfirst = np.array(
        [2, 1.8, 1.3, 0.7, 0.35, -0.1, -0.4, -0.5, -0.5, 0, 0.9, 0.95, 0.6, 0.2, -0.4, -0.8, -0.7, 0.2]) / 100
    Yfirst = np.array(
        [2, 1.8, 1.2, 0.3, -0.25, -0.7, -1.2, -1.8, -2, -2.5, -2.15, -1.3, -0.9, -0.5, 0.1, 1, 1.5, 1]) / 100
    Tfirst = np.array(
        [54.45, 54.53, 54.60, 54.65, 54.68, 54.7, 54.73, 54.75, 54.77, 54.79, 54.83, 54.87, 54.89, 54.92, 54.96, 55.02,
         55.09, 55.19]) - 54.45

    Xsecond = np.array([1.6, 1.2, 0.9, 0.7, 0.2]) / 100
    Ysecond = np.array([-2.2, -1.7, -1.4, -1.1, -0.7]) / 100
    Tsecond = -np.array([10.11, 10.05, 10.03, 10.01, 9.98]) + 10.11
    # [ 1.684e-02  3.384e-03  1.426e-01  1.812e-02 -2.435e-01]
    # Xforth=np.array([1,0.8,0.5,0.3,0.15,-0.2,-0.3,-0.5,-0.55,-0.35,-0.1,0.3,0.55,0.45,0.1,-0.25,-0.6,-0.85,-0.95,-0.9,-0.7,-0.3,0.2,0.45,0.8,1.05,1.25,1.25,0.95,-0.4,-0.75,-0.9,-0.75,-0.1,0.4,1,1.35,1.4,1.4,1.1,0.7,0.5,0.1,-0.2,-0.7,-1.15,-1.1,-1,-0.7])/100
    # Yforth=np.array([2,1.75,1.25,0.8,0.3,-0.4,-0.6,-1.3,-1.6,-2.3,-2.5,-2.25,-1.75,-1.25,-0.7,-0.3,0.4,0.9,1.35,1.7,1.7,1.45,0.9,0.75,0.2,-0.2,-0.8,-1.2,-1.5,-1.8,-2,-2,-1.65,-1.25,-1.2,-1,-0.75,-0.6,-0.1,0.2,0.6,0.65,0.95,1.2,1.3,1.1,0.9,0.5,0.15])/100
    # Tforth=np.array([1.86,1.92,1.96,1.98,2.01,2.03,2.05,2.08,2.1,2.13,2.16,2.18,2.20,2.23,2.26,2.29,2.32,2.36,2.39,2.44,2.47,2.51,
    #                    2.55,2.58,2.61,2.6,2.67,2.71,2.76,2.81,2.85,2.89,2.92,2.95,2.98,3.01,3.06,3.1,3.13,3.16,3.19,3.21,3.24,3.27,3.31,3.36,3.39,3.43,3.46])-1.86
    #----
    Xfive = np.array(
        [1.1, 0.75, 0.55, 0.3, -0.15, -0.5, -0.8, -1.25, -1.7, -1.95, -2, -1.8, -1.3, -0.75, -0.25, 0.4, 1.15, 1.65,
         1.85, 1.75, 1.5, 1.1, 0.55, -0.1, -0.7, -1.2, -1.5, -1.75, -1.7, -1.3, -0.8, -0.05, 0.45, 1.2, 1.45, 1.55, 1.6,
         1.35, 1.1, 0.85, 0.45, 0.05, -0.35, -0.65, -0.9, -1.1, -0.75]) / 100
    Yfive = np.array(
        [2.3, 1.75, 1.25, 0.7, -0.25, -0.75, -1.2, -1.35, -1.2, -0.75, -0.5, 0, 0.55, 1.05, 1.35, 1.45, 1.55, 1.35,
         0.85, 0.45, 0, -0.45, -0.8, -0.85, -0.7, -0.2, 0.35, 0.9, 1.45, 1.65, 1.6, 1.3, 0.85, 0.25, -0.25, -0.6, -0.9,
         -1.3, -1.25, -0.9, -0.4, 0.35, 1, 1.6, 1.9, 1.8, 1.25]) / 100
    Tfive = np.array(
        [13.87, 13.93, 13.97, 13.99, 14.03, 14.06, 14.09, 14.12, 14.15, 14.19, 14.21, 14.24, 14.28, 14.31, 14.34, 14.38,
         14.41, 14.45,
         14.49, 14.52, 14.55, 14.58, 14.61, 14.65, 14.68, 14.71, 14.74, 14.78, 14.83, 14.87, 14.9, 14.94, 14.97, 15.01,
         15.04, 15.07, 15.1, 15.14, 15.18, 15.21, 15.24, 15.28, 15.31, 15.35, 15.38, 15.45, 15.5]) - 13.87
    #[ 1.684e-02  3.384e-03  1.426e-01  1.812e-02 -2.435e-01]


    #++++++
    Xtherd = np.array(
        [2, 1.6, 1.1, 0.6, 0.2, -0.4, -0.9, -1.4, -1.8, -1.8, -1.4, -0.8, -0.15, 0.5, 1.1, 1.9, 2.1, 2, 1.5, 0.7, -0.3,
         -0.8, -1.3, -1.9, -2.2, -2.35, -2.2, -1.7, -0.9, 0.3, 1.1, 1.65, 1.8, 1.8, 1.5, 0.7, -0.1, -0.6, -1.4, -2, -2,
         -2.1, -1.8, -1.3, -0.35, 0.2]) / 100
    Ytherd = np.array(
        [2, 1.6, 1.2, 0.7, 0.3, -0.2, -0.7, -1.2, -1.7, -2.2, -2.1, -1.8, -1.5, -1.2, -0.8, -0.1, 0.5, 1, 1.4, 1.5, 1.3,
         1.2, 0.8, 0.4, -0.4, -0.8, -1.3, -1.5, -1.6, -1.5, -1.5, -1.35, -0.9, -0.5, 0.25, 0.8, 1.3, 1.5, 1.5, 1.3, 1.2,
         0.8, 0.1, -0.5, -1.3, -1.6]) / 100
    Ttherd = np.array(
        [50.46, 50.54, 50.58, 50.60, 50.63, 50.65, 50.67, 50.70, 50.73, 50.78, 50.83, 50.85, 50.87, 50.90, 50.92, 50.97,
         51, 51.05,
         51.09, 51.12, 51.17, 51.19, 51.21, 51.25, 51.27, 51.31, 51.35, 51.38, 51.42, 51.45, 51.49, 51.52, 51.56, 51.59,
         51.64, 51.67, 51.72, 51.74, 51.78, 51.83, 51.85, 51.89, 51.93, 51.97, 52.02, 52.04]) - 50.46
    # b=[ 3.580e-02  ,1.000e-08 , 1.196e-01 , 7.917e-01 , 8.298e-01]
def trajectory_diff_by_time(T1, X1, Y1, T2, X2, Y2,q):
    # Интерполяция X2 и Y2 как функций от T2
    interp_X2 = interp1d(T2, X2, kind='linear', fill_value="extrapolate")
    interp_Y2 = interp1d(T2, Y2, kind='linear', fill_value="extrapolate")

    # Значения X2, Y2 в моменты T1
    X2_at_T1 = interp_X2(T1)
    Y2_at_T1 = interp_Y2(T1)

    # Квадрат евклидова расстояния между точками в плоскости
    sq_distances = np.abs((X1 - X2_at_T1))**2+ np.abs((Y1 - Y2_at_T1))**2
    # Сумма квадратов (как ты просил)
    sum_sq_diff = np.sum(sq_distances)
    if q=='True':
        if max(T1)>max(T2)*1.2:
            return sum_sq_diff*10000,X2_at_T1,Y2_at_T1
        else:
            return sum_sq_diff,X2_at_T1,Y2_at_T1
    else:
        if max(T1) > max(T2) * 1.2:
            return sum_sq_diff * 10000
        else:
            return sum_sq_diff
def function(C,magnet,gamma1,gamma2,strong):
    phi=C[0]
    theta=C[1]
    x=C[2]
    y=C[3]
    xcoord = L * np.sin(theta) * np.cos(phi)
    ycoord = L * np.sin(theta) * np.sin(phi)
    zcoord = z0 + L - np.sqrt(L ** 2 - xcoord ** 2 - ycoord ** 2)
    F = np.array([float(0), float(0), float(0)])
    moment=strong*np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),-np.cos(theta)])
    for mcoordx,mcoordy, strength in magnet.magnets:
        dx = xcoord - mcoordx  # от магнита до точки стержня
        dy = ycoord - mcoordy
        dz = zcoord  # расстояние по вертикали до плоскости
        mu1 = np.array([0, 0, strength])
        r = np.array([dx, dy, dz])
        r2 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        r_hat=r/r2
        force_magnitude = ((3*mu0) / (4 * np.pi * r2 ** 4)) * (
                    moment * (r_hat @ mu1) + mu1 * (r_hat @ moment) + r_hat * (
                        mu1 @ moment) - 5 * r_hat * (r_hat @ moment) * (r_hat @ mu1))
        F+=force_magnitude
    Wx = F[1] * np.cos(phi) - F[0] * np.sin(phi)
    Wy = F[0] * np.cos(theta) * np.cos(phi) + F[1] * np.cos(theta) * np.sin(phi) + F[2] * np.sin(theta)
    Third=-2*x*y*np.cos(theta)/(np.sin(theta)) + (1/(L*A1*(np.sin(theta))))*Wx -(L*gamma1/A1)*x*np.sqrt(y**2 +x**2 *np.sin(theta)**2)-(gamma2/A1)*x
    Forth = x*x*np.sin(theta)*np.cos(theta) - ((g*B1)/(L*A1))*np.sin(theta) + (1/(L*A1))*Wy - (L*gamma1/A1)*y*np.sqrt(y**2 +x**2 *np.sin(theta)**2)-(gamma2/A1)*y
    return np.array([x,y,Third,Forth])
def Solvestep(Magnet,PHI,THETA,PHIdot,THETAdot,gamma1,gamma2,T,strong):
    thetanow = Magnet.quation()[1]
    h=h0* np.abs(thetanow)
    yn= Magnet.quation()
    k1 = function(yn,Magnet,gamma1,gamma2,strong)
    k2=function(yn+k1*h/2,Magnet,gamma1,gamma2,strong)
    k3=function(yn+k2*h/2,Magnet,gamma1,gamma2,strong)
    k4=function(yn+h*k3,Magnet,gamma1,gamma2,strong)
    ynew = yn+(h/6)*(k1+2*k2+2*k3+k4)
    PHI.append(ynew[0])
    THETA.append(ynew[1])
    PHIdot.append(ynew[2])
    THETAdot.append(ynew[3])
    Magnet.moove(ynew)
    T.append(T[-1]+h)
class Magnet:
    def __init__(self,theta0,phi0,x0,y0,Array):
        """
        Параметры маятника
        L - длина стержня
        m - масса магнита
        g - ускорение свободного падения
        damping - коэффициент демпфирования
        x- скорость по phi
        y-скорость по theta
        """

        self.theta=theta0
        self.phi=phi0
        self.y=y0
        self.x=x0
        self.magnets =Array  # список магнитов: [(x, y, strength), ...]
    def quation(self):
        return np.array([self.phi,self.theta,self.x,self.y])
    def moove(self,ynew):
        self.phi = ynew[0]
        self.theta = ynew[1]
        self.x = ynew[2]
        self.y = ynew[3]
def minimfunction(a):
    gamma1,gamma2,strong,x0,y0 = a
    theta0 = np.asin((0.023**2+0.011**2)**0.5/L)
    phi0 = np.arctan(2.3/1.1)
    Magnetic_position = [[0, -2 / 100, strong]]
    Mymagnit = Magnet(theta0=theta0, phi0=phi0, x0=x0, y0=y0, Array=Magnetic_position)
    PHI = [phi0]
    THETA = [theta0]
    T=[0]
    PHIdot=[x0]
    THETAdot=[y0]
    Nlen = 10000
    for i in range(Nlen):
        Solvestep(Mymagnit, PHI, THETA,PHIdot,THETAdot,gamma1,gamma2,T,strong)
    X = np.sin(THETA) * L * np.cos(PHI)
    Y = np.sin(THETA) * L * np.sin(PHI)
    Errorr =trajectory_diff_by_time(Tfive,Xfive,Yfive,T,X,Y,q='False')
    print(a,Errorr)
    return Errorr
#result = minimize(minimfunction, np.array([ 3.580e-02 ,0, 0.17,  0,  0]),bounds=((0,100),(0,100000),(0,10),(-1,1),(-1,1)))
#b = result.x
b= [0.03,  0,  0.13, 3,5]
Nlen = 20000

strong = b[2]
theta0 = np.asin((0.033 ** 2 + 0.011 ** 2) ** 0.5 / L)
phi0 = np.arctan(0.6 / 2)
Magnetic_position = [[-0.7/100,0.3/100,-strong],[-1/100,-0.4/100,strong],[1/100,0.5/100,strong],[-0.5/100,-1/100,-strong],[0.2/100,-2/100,strong],[0,0,strong]]

#print(result)
for k in range(1):
    # +;- - притяжение
    gamma1=b[0]
    gamma2=b[1]

    x0 = b[3]
    y0 =b[4]
    for i in range(len(Magnetic_position)):
        if Magnetic_position[i][2] > 0:
            plt.plot(Magnetic_position[i][0], Magnetic_position[i][1], 'o', color='g')
        else:
            plt.plot(Magnetic_position[i][0], Magnetic_position[i][1], 'o', color='red')
    Mymagnit = Magnet(theta0=theta0, phi0=phi0, x0=x0, y0=y0, Array=Magnetic_position)
    PHI1 = [phi0]
    THETA1 = [theta0]
    T = [0]
    PHIdot1=[x0]
    THETAdot1=[y0]
    for i in range(Nlen):
        Solvestep(Mymagnit, PHI1, THETA1,PHIdot1,THETAdot1,gamma1,gamma2,T,strong)
        if i % (Nlen / 10) == 0:
            print(100 * i // (Nlen), '%')
    X1 = np.sin(THETA1) * L * np.cos(PHI1)
    Y1 = np.sin(THETA1) * L * np.sin(PHI1)
    hyt,X_point,Y_point=trajectory_diff_by_time(Tfive,Xfive,Yfive,T,X1,Y1,q='True')
    for mnj in range(0):
        Texp=Tfive
        X=Xfive
        Y=Yfive
        print(cor(X_point, X))
        print(cor(Y,Y_point))
        Xerr=np.ones(len(X))*0.001
        Yerr=np.ones(len(Y))*0.001
        Texperr=np.ones(len(Texp))*0.02
        plt.plot(X1,Y1,color='r',alpha=0.4)
        plt.plot(X_point,Y_point,'o',color='r',label='Модель')
        plt.errorbar(X,Y,xerr=Xerr,yerr=Yerr,capsize=3,color='b',label='Экспериментальные данные.')
        plt.plot(X,Y,'o',color='b')
        plt.xlabel('X,м')
        plt.ylabel('Y,м')
        plt.grid()
        plt.legend()
        plt.show()

        plt.plot(T, X1,color='r',label='Модель')
        plt.errorbar(Texp,X,xerr=Texperr,yerr=Xerr,color='b',label='Экспериментальные данные.',capsize=3)
        plt.plot(Texp,X,'o',color='b')
        plt.xlabel('T,с')
        plt.ylabel('X,м')
        plt.legend()

        plt.grid()
        plt.show()


        plt.plot(T, Y1,color='r',label='Модель')
        plt.errorbar(Texp,Y,xerr=Texperr,yerr=Yerr,color='b',label='Экспериментальные данные.',capsize=3)
        plt.plot(Texp,Y,'o',color='b')
        plt.xlabel('T,с')
        plt.ylabel('Y,м')
        plt.grid()
        plt.legend()
        plt.show()

    print(max(T))
plt.plot(X1,Y1)
plt.grid()
plt.xlabel('X,м')
plt.ylabel('Y,м')
plt.show()
Full=[]
SHTRIHI = []
for k in range(1):
    # +;- - притяжение
    gamma1=b[0]
    gamma2=b[1]

    x0 = b[3]
    y0 =b[4]*(1+k/1000)
    for i in range(len(Magnetic_position)):
        if Magnetic_position[i][2] > 0:
            plt.plot(Magnetic_position[i][0], Magnetic_position[i][1], 'o', color='g')
        else:
            plt.plot(Magnetic_position[i][0], Magnetic_position[i][1], 'o', color='orange')
    Mymagnit = Magnet(theta0=theta0, phi0=phi0, x0=x0, y0=y0, Array=Magnetic_position)
    PHIdot=[x0]
    THETAdot=[y0]
    PHI = [phi0]
    THETA = [theta0]
    T = [0]
    for i in range(Nlen):
        Solvestep(Mymagnit, PHI, THETA,PHIdot,THETAdot,gamma1,gamma2,T,strong)
        if i % (Nlen / 10) == 0:
            print(100 * i // (Nlen), '%')
    X = np.sin(THETA) * L * np.cos(PHI)
    Y = np.sin(THETA) * L * np.sin(PHI)
    Xdot = L*np.sin(THETA)*np.cos(PHI)*THETAdot - L *np.sin(THETA)*np.sin(PHI)*PHIdot
    Ydot = L*np.sin(THETA)*np.sin(PHI)*THETAdot+L *np.sin(THETA)*np.cos(PHI)*PHIdot
    Full.append([T,X,Y])
    SHTRIHI.append([T,Xdot,Ydot])
animate_trajectories_without_tail(Full,magnets=Magnetic_position,duration=max(Full[0][0]),fps=60)



for i in range(0):
    plt.show()
    gdffg, X_sec, Y_sec= trajectory_diff_by_time(Full[0][0], Full[0][1], Full[0][2], Full[1][0], Full[1][1],
                                                  Full[1][2], q='True')
    gdffjg, X_sec_dot, Y_sec_dot= trajectory_diff_by_time(SHTRIHI[0][0], SHTRIHI[0][1], SHTRIHI[0][2], SHTRIHI[1][0], SHTRIHI[1][1],
                                                  SHTRIHI[1][2], q='True')
    X_sec = np.array(X_sec)
    Y_sec = np.array(Y_sec)
    X_fis_dot=np.array(SHTRIHI[0][1])
    Y_fis_dot = np.array(SHTRIHI[0][2])
    X_fis = np.array(Full[0][1])
    Y_fis = np.array(Full[0][2])
    r = (np.sqrt((X_fis - X_sec) ** 2 + (Y_fis - Y_sec) ** 2))
    r_dot = (np.sqrt((X_fis_dot - X_sec_dot) ** 2 + (Y_fis_dot - Y_sec_dot) ** 2))
    r_final = np.sqrt(r_dot**2+r**2)/(np.abs(b[4])/1000)
    plt.plot(Full[0][0], np.log(r_final), color='green',label='Сгенерированные данные.')
    plt.grid()
    plt.xlabel('T,с')
    plt.ylabel(r'$\ln(\frac{\delta q}{\delta q_0})$')

    k, b, r = aprocsimate(Full[0][0][:int(Nlen*2.35/max(Full[0][0]))], np.log(r_final)[:int(Nlen*2.35/max(Full[0][0]))])
    print(k, b, r)
    plt.plot(Full[0][0], k * np.array(Full[0][0]) + b, color='red',label='Аппроксимация')
    plt.show()


