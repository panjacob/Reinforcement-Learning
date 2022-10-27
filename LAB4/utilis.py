from math import pi
import numpy as np


def best_action(Q):
    pass


RESOLUTION = 100
bin_state0 = np.linspace(-10, 10, num=RESOLUTION)
bin_state1 = np.linspace(-10, 10, num=RESOLUTION)
bin_state2 = np.linspace(-10, 10, num=RESOLUTION)
bin_state3 = np.linspace(-10, 10, num=RESOLUTION)
bin_F = np.linspace(-10, 10, num=RESOLUTION)


def encode_Q(state, F):
    result = np.zeros(5)


# Obliczenie stanu wahadla w kolejnym kroku czasowym metoda analityczna
# stan - wektor parametrow stanu w czasie t
# stann -   -||- w czasie t + dt
# F - sila dzialajaca na wozek
def wahadlo(stan, F):
    Fmax, krokcalk, g, tar, masawoz, masawah, drw = wah_glob()

    if F > Fmax:
        F = Fmax
    if F < -Fmax:
        F = -Fmax

    hh = krokcalk * 0.5;
    momwoz = masawoz * drw;
    momwah = masawah * drw;
    cwoz = masawoz * g;
    cwah = masawah * g;

    sx = np.sin(stan[0]);
    cx = np.cos(stan[0]);
    c1 = masawoz + masawah * sx * sx;
    c2 = momwah * stan[1] * stan[1] * sx;
    c3 = tar * stan[3] * cx;

    stanpoch = np.zeros(stan.size)

    stanpoch[0] = stan[1];
    stanpoch[1] = ((cwah + cwoz) * sx - c2 * cx + c3 - F * cx) / (drw * c1);
    stanpoch[2] = stan[3];
    stanpoch[3] = (c2 - cwah * sx * cx - c3 + F) / c1;
    stanh = np.zeros(stan.size)
    for i in range(4):
        stanh[i] = stan[i] + stanpoch[i] * hh;

    sx = np.sin(stanh[0]);
    cx = np.cos(stanh[0]);
    c1 = masawoz + masawah * sx * sx;
    c2 = momwah * stanh[1] * stanh[1] * sx;
    c3 = tar * stanh[3] * cx;

    stanpochh = np.zeros(stan.size)
    stanpochh[0] = stanh[1];
    stanpochh[1] = ((cwah + cwoz) * sx - c2 * cx + c3 - F * cx) / (drw * c1);
    stanpochh[2] = stanh[3];
    stanpochh[3] = (c2 - cwah * sx * cx - c3 + F) / c1;
    stann = np.zeros(stan.size)
    for i in range(4):
        stann[i] = stan[i] + stanpochh[i] * krokcalk;
    if stann[0] > np.pi:
        stann[0] = stann[0] - 2 * pi;
    if stann[0] < -np.pi:
        stann[0] = stann[0] + 2 * pi;

    return stann


def wah_glob():
    Fmax = 1000
    krokcalk = 0.05
    g = 9.8135
    tar = 0.02
    masawoz = 10
    masawah = 20
    drw = 20
    return Fmax, krokcalk, g, tar, masawoz, masawah, drw


def wahadlo_test(stanp):
    Fmax, krokcalk, g, tar, masawoz, masawah, drw = wah_glob()
    pli = open('historia.txt', 'w')
    pli.write("Fmax = " + str(Fmax) + "\n")
    pli.write("krokcalk = " + str(krokcalk) + "\n")
    pli.write("g = " + str(g) + "\n")
    pli.write("tar = " + str(tar) + "\n")
    pli.write("masawoz = " + str(masawoz) + "\n")
    pli.write("masawah = " + str(masawah) + "\n")
    pli.write("drw = " + str(drw) + "\n")

    sr_suma_nagrod = 0
    liczba_krokow = 0
    liczba_stanow_poczatkowych, lparam = stanp.shape
    for epizod in range(liczba_stanow_poczatkowych):
        # Wybieramy stan poczatkowy:
        # stan = rand(1,4).*[pi/1.5 pi/1.5 20 20] - [pi/3 pi/3 10 10]; % met.losowa
        nr_stanup = epizod
        stan = stanp[nr_stanup, :]

        krok = 0
        suma_nagrod_epizodu = 0
        czy_przewrocenie_wahadla = 0
        while (krok < 1000) & (czy_przewrocenie_wahadla == 0):
            krok = krok + 1

            # Wyznaczamy akcje a (sile) w stanie stan zgodnie z wyuczona strategia
            # (bez eksploracji)
            # ........................................................
            # ........................................................
            F = 0

            # wyznaczenie nowego stanu:
            nowystan = wahadlo(stan, F)

            czy_przewrocenie_wahadla = (abs(nowystan[0]) >= np.pi / 2)
            R = nagroda(stan, nowystan, F)
            suma_nagrod_epizodu = suma_nagrod_epizodu + R

            pli.write(str(epizod + 1) + "  " + str(stan[0]) + "  " + str(stan[1]) + "  " + str(stan[2]) + "  " + str(
                stan[3]) + "  " + str(F) + "\n")

            stan = nowystan

        sr_suma_nagrod = sr_suma_nagrod + suma_nagrod_epizodu / liczba_stanow_poczatkowych
        liczba_krokow = liczba_krokow + krok
        print("w %d epizodzie suma nagrod = %g, liczba krokow = %d" % (epizod, suma_nagrod_epizodu, krok))

    print("srednia suma nagrod w epizodzie = %g" % (sr_suma_nagrod))
    print("srednia liczba krokow ustania wahadla = %g" % (liczba_krokow / liczba_stanow_poczatkowych))

    pli.close()


def nagroda(stan, nowystan, F):
    kara_za_odchylenie = nowystan[0] ** 2 + 0.25 * nowystan[1] ** 2 + 0.0025 * nowystan[2] ** 2 + 0.0025 * nowystan[
        3] ** 2
    kara_za_przewrocenie = (abs(nowystan[0]) >= np.pi / 2) * 1000
    # ..............................................
    # ..............................................
    return -(kara_za_odchylenie + kara_za_przewrocenie)
