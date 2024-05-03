import matplotlib.pyplot as plt
import numpy as np

from chi_square import ChiSquaredDistribution
from exponential import ExponentialDistribution
from normal import NormalDistribution
from student import StudentDistribution
from uniform import UniformDistribution
from weibull import WeibullDistribution


def main():
    np.random.seed(3)
    print('Дискретные распределения\nМеню')
    print('Выберите распределение\n'
          '1 - равномерное распределение\n'
          '2 - нормальное распределение\n'
          '3 - экспоненциальное распределение\n'
          '4 - хи-квадрат распределение\n'
          '5 - распределение Стьюдента\n'
          '6 - распределение Вейбулла')
    distribution_key = int(input())
    match distribution_key:
        case 1:
            print('Введите параметры распределения\n'
                  'low = ', end='')
            low = float(input())
            print('high = ', end='')
            high = float(input())
            print('size = ', end='')
            size = int(input())
            uniform = UniformDistribution(low=low, high=high, size=size)
            M, D = uniform.mean(), uniform.var()
            print(f'Мат. ожидание и дисперсия соответственно: M = {M}, D = {D}')
            print(f'Отклонения от теор. значений: M: {abs(M - uniform.mean_T())}, D: {abs(D - uniform.var_T())}')
        case 2:
            print('Введите параметры распределения\n'
                  'm = ', end='')
            m = float(input())
            print('sigma = ', end='')
            sigma = float(input())
            print('size = ', end='')
            size = int(input())
            normal = NormalDistribution(mu=m, sigma=sigma, size=size)
            M, D = normal.mean(), normal.var()
            print(f'Мат. ожидание и дисперсия соответственно: M = {M}, D = {D}')
            print(f'Отклонения от теор. значений: M: {abs(M - normal.mean_T())}, D: {abs(D - normal.var_T())}')
        case 3:
            print('Введите параметры распределения\n'
                  'lambda = ', end='')
            scale = float(input())
            print('size = ', end='')
            size = int(input())
            exponential = ExponentialDistribution(scale=scale, size=size)
            M, D = exponential.mean(), exponential.var()
            print(f'Мат. ожидание и дисперсия соответственно: M = {M}, D = {D}')
            print(
                f'Отклонения от теор. значений: M: {abs(M - exponential.mean_T())}, D: {abs(D - exponential.var_T())}')
        case 4:
            print('Введите параметры распределения\n'
                  'N = ', end='')
            N = int(input())
            print('size = ', end='')
            size = int(input())
            chi2 = ChiSquaredDistribution(N=N, size=size)
            M, D = chi2.mean(), chi2.var()
            print(f'Мат. ожидание и дисперсия соответственно: M = {M}, D = {D}')
            print(f'Отклонения от теор. значений: M: {abs(M - chi2.mean_T())}, D: {abs(D - chi2.var_T())}')
        case 5:
            print('Введите параметры распределения\n'
                  'N = ', end='')
            N = int(input())
            print('size = ', end='')
            size = int(input())
            s = StudentDistribution(N=N, size=size)
            M, D = s.mean(), s.var()
            print(f'Мат. ожидание и дисперсия соответственно: M = {M}, D = {D}')
            print(f'Отклонения от теор. значений: M: {abs(M - s.mean_T())}, D: {abs(D - s.var_T())}')
        case 6:
            print('Введите параметры распределения\n'
                  'lambda = ', end='')
            scale = float(input())
            print('m = ', end='')
            m = float(input())
            print('size = ', end='')
            size = int(input())
            wb = WeibullDistribution(scale=scale, size=size, m=m)
            M, D = wb.mean(), wb.var()
            print(f'Мат. ожидание и дисперсия соответственно: M = {M}, D = {D}')
            print(f'Отклонения от теор. значений: M: {abs(M - wb.mean_T())}, D: {abs(D - wb.var_T())}')
            plt.hist(wb.val, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel('Значения выборки')
            plt.ylabel('Частота')
            plt.title('Гистограмма распределения выборки')
            plt.savefig('hist.png')
            plt.show()
            from scipy.stats import weibull_min
            from scipy.stats import chi2

            n = size
            k = m
            lam = scale

            x = weibull_min.rvs(k, loc=0, scale=lam, size=n)
            print('Рассчитываем значения статистики критерия Пирсона')
            chi_square_statistic = sum((x - wb.val) ** 2 / x)
            print(chi_square_statistic)
            critical_value = chi2.ppf(0.9, len(wb.val) - 1)
            print(f'Критическое значение {critical_value}')
            if chi_square_statistic < critical_value:
                print("Нулевая гипотеза принимается")
            else:
                print("Нулевая гипотеза отвергается")


if __name__ == '__main__':
    main()
