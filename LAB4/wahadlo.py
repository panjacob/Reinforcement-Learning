from random import random

from numpy.random import rand

from utilis import *


def wahadlo_uczenie():
    # liczba_epizodow = 100_000_000
    liczba_epizodow = 10_000
    alfa = 0.001  # wsp.szybkosci uczenia(moze byc funkcja czasu)
    epsilon = 0.5  # wsp.eksploracji(moze byc funkcja czasu)

    stanp = np.array(
        [[np.pi / 6, 0, 0, 0], [0, np.pi / 3, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10], [np.pi / 12, np.pi / 6, 0, 0],
         [np.pi / 12, -np.pi / 6, 0, 0], [-np.pi / 12, np.pi / 6, 0, 0], [-np.pi / 12, -np.pi / 6, 0, 0],
         [np.pi / 12, 0, 0, 0], [0, 0, -10, 10]], dtype=float)
    liczba_stanow_poczatkowych, lparam = stanp.shape

    # inicjacja kodowania, wyznaczenie liczby parametrów (wag):
    # ........................................................
    # ........................................................

    # inicjacja wektora wag:
    liczba_wag = 1000  # na razie, by sie uruchomilo
    w = np.zeros(liczba_wag)

    Q = np.zeros([RESOLUTION, RESOLUTION, RESOLUTION, RESOLUTION, RESOLUTION], dtype=float)

    for epizod in range(liczba_epizodow):
        # Wybieramy stan poczatkowy:
        # stan = rand(1,4).*[pi/1.5 pi/1.5 20 20] - [pi/3 pi/3 10 10]; % met.losowa
        nr_stanup = epizod % liczba_stanow_poczatkowych
        stan = stanp[nr_stanup, :]

        krok = 0
        czy_przewrocenie_wahadla = 0
        while (krok < 1000) & (czy_przewrocenie_wahadla == 0):
            krok = krok + 1

            # Wyznaczamy akcje a (sile) w stanie stan z uwzględnieniem
            # eksploracji (np. metoda epsilon-zachlanna lub softmax)
            # ........................................................
            # ........................................................
            # Epsilon greedy
            if random() > epsilon:
                F = np.random.uniform(0, 2)
            else:
                F = best_action()

            # wyznaczenie nowego stanu:
            nowystan = wahadlo(stan, F)
            print('nowy stan', nowystan)

            czy_przewrocenie_wahadla = (abs(nowystan[0]) >= np.pi / 2)
            R = nagroda(stan, nowystan, F)

            # Aktualizujemy wartosci Q dla aktualnego stanu i wybranej akcji:
            # ........................................................
            # ........................................................
            # w = w + ...

            stan = nowystan

        # co jakis czas test z wygenerowaniem historii do pliku:
        if epizod % 1000 == 0:
            print((epizod / liczba_epizodow) * 100, "%")
            wahadlo_test(stanp)


wahadlo_uczenie()
