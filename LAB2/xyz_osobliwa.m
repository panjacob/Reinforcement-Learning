% skrypt do przykladu XYZ z MDP
% szukanie optymalnej strategii metod¹ programowania dynamicznego z iteracj¹
% strategii oraz rozwiazywaniem ukladu rownan liniowych w celu wyznaczenia
% uzytecznosci stanow V

clear

gamma = 0.99

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

p = zeros(liczba_stanow,liczba_stanow,liczba_akcji);
r = zeros(liczba_stanow,liczba_stanow);

p(X,X,A1) = 0.65;
p(Y,X,A1) = 0.35;
p(Z,X,A1) = 0.0;

p(X,X,A2) = 0.2;
p(Y,X,A2) = 0.8;
p(Z,X,A2) = 0.0;

p(X,X,A3) = 0.25;
p(Y,X,A3) = 0.75;
p(Z,X,A3) = 0.0;

p(X,Y,A4) = 1;
p(Z,Y,A4) = 0;

p(X,Y,A5) = 1;
p(Z,Y,A5) = 0;

r(X,Y) = 1;
r(X,Z) = 0.8;
r(Y,Z) = 1;

aX = 1;
aY = 1;
zmiana_strategii = 1;
liczba_zmienionych_akcji  = liczba_stanow;

while liczba_zmienionych_akcji > 0
    
    % ---------------------------------------------------------------------------------------------   
    % ----------------------  1. Oblicznie uzytecznosci dla zadanej strategii  -------------------- 
    % --------------------------------------------------------------------------------------------- 
    liczba_zmienionych_akcji = 0;
    P = [p(:,X,aX)'; p(:,Y,aY)'; p(:,Z,1)'] % mac. prawdop. przejsc
    M = eye(liczba_stanow) - gamma*P     % macierz przed wartosciami stanu   
    rsa = sum(r.*P,2)                      % wektor srednich nagrod za wykonanie akcji w stanie
    wyznacznik = det(M)
    
    if (wyznacznik == 0)
        disp(sprintf('Nie da sie policzyc uzytecznosci - wyznacznik = 0'));
        break;
    end
    
    V = inv(M)*rsa               % uzytecznosci stanow

    % uzytecznosci akcji:
    QX = zeros(1,liczby_akcji(X));
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
    if (aX_max ~= aX)&&(QX_max > QX(aX))
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
end

disp(sprintf('strategia optymalna: w stanie X akcja %s, w stanie Y akcja %s',akcjeX{aX},akcjeY{aY})); 

