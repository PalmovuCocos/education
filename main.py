import numpy as np

def df(x):  # функция вычисления производной    (Поменять)
    return 0.5 * (1+x) * (1-x)


def act(x): # пороговая функция активации
    return 0 if x < 0.5 else 1

w1 = np.array([-0.2, 0.3])

# функция пропуска вектора наблюдений через НС
# тут запоминаются выходы нейрона
# мне данная функция не нужна, т.к. нет скрытых слоев, можно использовать функцию нейронки
#def go_forward(inp):
#    sum = np.dot(w1, inp)
#    out = np.array([act(x) for x in sum])
#    y = act(sum)
#    return (y, out)

def train(epoch):
    global w1
    lmd = 0.01
    N = 10000
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]
        y = log_or(x[0], x[1])
        e = y - x[-1]
        delta = e * df(y)
        w1[0] = w1[0] - np.array(x[0]) * delta * lmd
        w1[1] = w1[1] - np.array(x[1]) * delta * lmd


def log_or(operand1, operand2):
    x = np.array([operand1, operand2])
    w1 = np.array([0.5, 0.5])
    sum = np.dot(w1, x)
    #print(sum,'сумма на нейроне')
    y = act(sum)
    return y

def log_and(operand1, operand2):
    x = np.array([operand1, operand2])
    w1 = np.array([1, 1])
    sum = np.dot(w1, x) - 1.5
    #print(sum,'сумма на нейроне')
    y = act(sum)
    return y

epoch_or = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1)
]
epoch_and = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]
#train(epoch_or)    # запуск обучения сети

# проверка полученных результатов
#for x in epoch_or:
#    y = log_or(x[0],x[1])
#    print(f"Выходное значение НС: {y} => {x[-1]}")



train(epoch_and)    # запуск обучения сети
for x in epoch_and:
    y = log_and(x[0],x[1])
    print(f"Выходное значение НС: {y} => {x[-1]}")
#print(log_or(0,0))
#print(log_and(1,1))

