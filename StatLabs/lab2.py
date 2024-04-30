import numpy as np
import scipy


def distribution_parameters(distribution):
    M = distribution.sum() / distribution.size
    D = ((distribution - M) ** 2).sum() / distribution.size
    return M, D


def count_ones_zeroes(bits):
    ones = sum(map(int, bits))
    return len(bits) - ones, ones


def monobit_test(bits):
    n = len(bits)

    zeroes, ones = count_ones_zeroes(bits)
    s = abs(ones - zeroes)
    print("  Ones count   = %d" % ones)
    print("  Zeroes count = %d" % zeroes)

    p = scipy.special.erfc(s / (np.sqrt(n) * np.sqrt(2.0)))

    success = (p >= 0.01)
    return success, p


if __name__ == '__main__':
    np.random.seed(3)
    print('Дискретные распределения\nМеню')
    print('Выберите распределение\n'
          '1 - равномерное распределение\n'
          '2 - биномиальное распределение\n'
          '3 - геометрическое распределение\n'
          '4 - распределение Пуассона\n'
          '5 - тест NIST')
    distribution_key = int(input())
    match distribution_key:
        case 1:
            print('Введите параметры распределения\n'
                  'low = ', end='')
            low = int(input())
            print('high = ', end='')
            high = int(input())
            print('size = ', end='')
            size = int(input())
            uniform = np.random.randint(low=low, high=high, size=size)
            print(f'Мат. ожидание и дисперсия соответственно: M = {distribution_parameters(uniform)[0]:.2f}, D = {distribution_parameters(uniform)[1]:.2f}')
        case 2:
            print('Введите параметры распределения\n'
                  'n = ', end='')
            n = int(input())
            print('p = ', end='')
            p = int(input())
            print('size = ', end='')
            size = int(input())
            binomial = np.random.binomial(n, p)
            print(f'Мат. ожидание и дисперсия соответственно: M = {distribution_parameters(binomial)[0]:.2f}, D = {distribution_parameters(binomial)[1]:.2f}')
        case 3:
            print('Введите параметры распределения\n'
                  'p = ', end='')
            p = int(input())
            print('size = ', end='')
            size = int(input())
            geometric = np.random.geometric(p, size)
            print(f'Мат. ожидание и дисперсия соответственно: M = {distribution_parameters(geometric)[0]:.2f}, D = {distribution_parameters(geometric)[1]:.2f}')
        case 4:
            print('Введите параметры распределения\n'
                  'lambda = ', end='')
            lam = int(input())
            print('size = ', end='')
            size = int(input())
            poisson = np.random.poisson(lam, size)
            print(f'Мат. ожидание и дисперсия соответственно: M = {distribution_parameters(poisson)[0]:.2f}, D = {distribution_parameters(poisson)[1]:.2f}')
        case 5:
            print('Введите строку, содержащую только 0 или 1. Любой символ, отличный от 1, считается 0')
            s = input()
            success, p = monobit_test(s)
            if success:
                print('Тест пройден, p =', p)
            else:
                print('Тест не пройден, p =', p)
