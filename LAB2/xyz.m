% skrypt do przykladu XYZ z MDP
% szukanie optymalnej strategg metod¹ programowania dynamicznego z iteracj¹
% strategii oraz rozwiazywaniem ukladu rownan liniowych w celu wyznaczenia
% uzytecznosci stanow V

clear

gamma = 0.8               % wspolczynnik dyskontowania

disp(sprintf('szukanie optymalnej strategii met. TD'))

X = 1;
Y = 2;
Z = 3;
stany = [X Y Z]

akcjeX = {'A1' 'A2' 'A3'};
akcjeY = {'A4' 'A5'};

liczby_akcji = [3 2 0];   % liczby akcji dla poszczegolnych stanow    

liczba_akcji = max(liczby_akcji);
liczba_stanow = length(stany);

A1=1; A2 = 2; A3 = 3; A4 = 1; A5 = 2;

p = zeros(liczba_stanow,liczba_stanow,liczba_akcji);     % prawdopodobienstwa przejœæ pomiedzy stanami w zaleznosci od akcji
r = zeros(liczba_stanow,liczba_stanow);                  % nagrody za przejscia od stanu do stanu

p(X,X,A1) = 0.5;      % P(X|X,A1) 
p(Y,X,A1) = 0.2;      % P(Y|X,A1) - prawdopodobienstwo przejscia ze stanu X do stanu Y po wykonaniu akcji A1 
p(Z,X,A1) = 0.3;

p(X,X,A2) = 0.0;
p(Y,X,A2) = 0.5;
p(Z,X,A2) = 0.5;

p(X,X,A3) = 0.1;
p(Y,X,A3) = 0.6;
p(Z,X,A3) = 0.3;

p(X,Y,A4) = 0.4;
p(Z,Y,A4) = 0.6;

p(X,Y,A5) = 0.7;
p(Z,Y,A5) = 0.3;

r(X,Z) = 1;           % nagroda za przejscie ze stanu X do stanu Z
r(Y,Z) = 1;

aX = 1;               % numery akcji dla poczatkowej strategii
aY = 1;

liczba_zmienionych_akcji  = liczba_stanow;    % liczba stanow, dla których zmienily sie akcje podczas modyfikacji strategii

% glowna petla iteracji strategii:
while liczba_zmienionych_akcji > 0
    
    % ---------------------------------------------------------------------------------------------   
    % ----------------------  1. Oblicznie uzytecznosci dla zadanej strategii  -------------------- 
    % --------------------------------------------------------------------------------------------- 
    liczba_zmienionych_akcji = 0;
    P = [p(:,X,aX)'; p(:,Y,aY)'; p(:,Z,1)']    % mac. prawdop. przejsc
    M = eye(liczba_stanow) - gamma*P           % macierz wspolczynnikow wartosci stanow   
    rsa = sum(r.*P,2)                          % wektor srednich nagrod za wykonanie akcji w stanie
    wyznacznik = det(M)                        % wyznacznik macierzy
    
    if (wyznacznik == 0)
        disp(sprintf('Nie da sie policzyc uzytecznosci - wyznacznik = 0'));
        break;
    end
    
    V = inv(M)*rsa                             % uzytecznosci stanow

    % uzytecznosci akcji:
    QX = zeros(1,liczby_akcji(X));             % wektor uzytecznosci akcji w stanie X
    for i=1:liczby_akcji(X)
        a = i;
        QX(i) = sum(p(:,X,a).*(r(X,:)' + gamma*V));
    end
    QY = zeros(1,liczby_akcji(Y));
    for i=1:liczby_akcji(Y)
        a = i;
        QY(i) = sum(p(:,Y,a).*(r(Y,:)' + gamma*V));
    end
    
    disp(sprintf('aktualna strategia: w stanie X akcja %s (Q = %f), w stanie Y akcja %s (Q = %f)',akcjeX{aX},QX(aX),akcjeY{aY},QY(aY))); 
    
    % ---------------------------------------------------------------------------------------------   
    % ----------------------  2. Aktualizacja strategii na podstawie uzytecznosci akcji  ---------- 
    % --------------------------------------------------------------------------------------------- 
    [QX_max, aX_max] = max(QX);
    if (aX_max ~= aX)&&(QX_max > QX(aX))       % jesli akcja nienalezaca do aktualnej strategii ma wieksza uzytecznosc
        disp(sprintf('w stanie X akcja %s o wart. %f lepsza od akcji ze strategii %s o wart. %f',akcjeX{aX_max},QX_max,akcjeX{aX},QX(aX)));
        aX = aX_max;
        liczba_zmienionych_akcji = liczba_zmienionych_akcji + 1;
    end

    [QY_max, aY_max] = max(QY);
    if (aY_max ~= aY)&&(QY_max > QY(aY))
        disp(sprintf('w stanie Y akcja %s o wart. %f lepsza od akcji ze strategii %s o wart. %f',akcjeY{aY_max},QY_max,akcjeY{aY},QY(aY)));
        aY = aY_max;
        liczba_zmienionych_akcji = liczba_zmienionych_akcji + 1;
    end
end  % petla iteracji strategii

disp(sprintf('strategia optymalna: w stanie X akcja %s, w stanie Y akcja %s',akcjeX{aX},akcjeY{aY})); 

