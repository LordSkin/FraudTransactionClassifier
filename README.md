# FraudTransactionClassifier

## Wstęp
Projekt ma na celu stworzenie kalsyfikatora, który wykrywa oszustwa wśród transakcji kartami kredytowymi. Do projektu wykorzystano zbiór danych znajdujący się pod adresem:
https://www.kaggle.com/mlg-ulb/creditcardfraud

Przedstawiony zbiór danych zawiera 284 807 transakcji, z czego 492 są zakwalifikowane jako oszustwo. Zbiór danych jest więc skrajnie niezbalansowny.
Każdy wpis zawiera 30 zanimizowanych cech oraz przynalezność do klasy (0,1)

## Wybór modelu
Wyuczono i przetestowano nastepujące modele:
* Sieć neuronowa z jedną warstwą ukrytą
* Las losowy
* Naiwny klasyfikator Bayesa
* Kwadratowa analiza dyskryminacyjna
* Regresja logistyczna
* KNN
* Support vector classifier

Następnie porównywano wyniki false-positive oraz false-negative na podstawie macierzy błędu.

#### Wnioski:
Klasyfikator powinien przede wszystkim dążyć do zminimmalizowania błędów false-negative czyli jak najrzadziej klasyfikować oszustwa jako prawidłowe transakcje, ale nie powinien też klasyfikować wszystkiego jako oszustwo.

Najmniejszy błąd false-negative osiągnęły naiwny klasyfikator Bayesa i kwadratowa analiza dyskryminacyjna, ale oba dają błąd false-positive kilka rzędów wyższy niż pozostałe algorytmy. Analizując błędy, oraz mając na uwadze liczbę hiperparametrów do zbadania wybrano **sieć neuronową**. Szczegółowe dane z testu znajdują się w notatniku *Wybor_modelu.ipynb*

## Wybór metody oversamplingu
Ponieważ dane są skrajnie niezbalansowane zdecydowano się chociaż częściowo naprawić ten problem poprzez oversampling istniejących próbek. Wszystkie modele na tym etapie były porównywane poprzez wyniki wyuczonych na nich sieci neuronowych po wygenerowaniu danych do poziomu wyrównania liczności klas.

Przetestowano metody:
* Brak oversamplingu
* SMOTE
* ADASYN
* SMOTE w połączeniu z TOMEk Links

#### Wnioski:
Wszystkie metody okazały się znacząco podnosić skoteczność w wykrywaniu fałszywych transakcji, różnice pomiędzy nimi samymi są jednak niewielkie. Zdecydowano się na uzycie moetody ***SMOTE***.

Szczegółowe testy znajdują się w pliku *Wybor_metody_oversamplingu.ipynb*.

## Wybór współczynnika oversamplingu
Współczynnik oversamplingu to procent licznośći jaki klasa stanowiąca mniejszość pierwotnie będzie stanowić w zbiorze po oversamplingu. Np. dla n = 0.7 W finalnym zbiorze 7 na 10 próbek będzie należało do klasy mniejszościowej(w tym wypadku są to transakcje fałszywe).

Testy przeprowadzono podobnie do poprzednich przypadków, trenując sieci na poszczególnych przypadkach i porównując ich wyniki. Testowano wartości:
* n = 0.4
* n = 0.5
* n = 0.65
* n = 0.75
* n = 0.85
* n = 0.9
* n = 0.95
* n = 0.99
* n = 0.999

#### Wnioski:
Wyniki wydają się zadziwiająco zbliżone, co więcej nie mozna znaleść wśród nich żadnej tendencji. Wybrano najkorzystniejszy wynik **n = 0.85**

Szczegółowe testy znajdują się w pliku *Wybor_wspolczynnika_oversamplingu.ipnb*.

## Wybór metody redukcji wymiarów
Dane wejściowe posiadają 30 wymiarów ilościowych. Większość z nich prawdopodobnie układa się w rozkład normalny. W redukcji testowano zarówno metody rzutujące wymiary na nowe, ale równierz metody bezpośrednio wybierające z istniejących wymiarów. Dla testów przyjęto wybór 15 wymiarów dla każdej z metod, i próg threshold = 0.8 dla metody progu wariancji.

Testowano metody:
* PCA
* Feature aglomeration
* Wybór K najlepszych
* Próg wariancji
* Recursive feature elimination
* Gaussian random projection

#### Wnioski:
Różnice wyników poszczególnych metod nie są spektakularne. Wybrano najlepszą metodę Wyboru K najlepszych cech.

Szczegółowe testy znajdują się w pliku *Wybor_metody_redukcji_wymiarow.ipnb*.
 
## Wybór liczby wymiarów
W poprzednim teście testowano poszczególne metody dla reduukcji do 15 wymiarów, liczbę te można jednak zmienią aby poprawić jakośc klasyfikacji lub przyspieszyć proces uczenia.
Testowano następujące liczby cech po zredukowaniu:
* k = 30 (bez redukcji)
* k = 25
* k = 20
* k = 15
* k = 10
* k = 5
* k = 3
* k = 2
* k = 1

#### Wnioski:
Przy redukcji wymiarów mozna zaobserwować zalezność, że przy zmniejszaniu liczby wymiarów od około k = 10 skuteczność klasyfikacji zayczna spadać. Nie są to jednak znaczne wartości. Dlatego zdecydowano się na redukcjędo 10 wymiarów (k = 10)

Szczegółowe testy znajdują się w pliku *Wybor_liczby_cech.ipnb*.

## Wybór funkcji aktywacji
Funkcja aktywacji jest wewnętrznym hiperparametrem sieci neuronowej.

Testowano funkcje:
* funkcja liniowa
* logistyczna sigmoida f(x) = 1 / (1 + exp(-x)).
* tangens hiperboliczny
* relu

#### Wnioski:
Najmniej błędów false-negative daje funkcja liniowa, jednak powoduje też nieproporcjkonalnie dużą liczbę błędów false-positive, dlatego wybrano trochę gorszą funkcję logistyczną sigmoidę.

Szczegółowe testy znajdują się w pliku *Wybor_funkcji_aktywacji.ipnb*.
 
## Wybór optymalizatora
Optymalizator, podobnie jak funkcja aktywacji jest wewnętrznym hiperparametrem sieci neuronowej.

Testowano metody:
* Gradient stochastyczny
* ADAM
* algorytm Broyden–Fletcher–Goldfarb–Shanno

#### Wnioski:
Najlepsze wyniki dla false-negative daje Gradient stochastyczny, jednak róznice te nie są zbyt duże. Za to w przypadku false-positive ADAM wydaje się stanowczo lepszy od pozostałych. Przy uzyciu optymalizacji ADAM sieć uczy się szybciej niż z SGD, dlatego zastosowano optymalizacje ADAM.

Szczegółowe testy znajdują się w pliku *Wybor_optymalizatora.ipnb*.

## Wybór liczby neuronów w warstwie ukrytej sieci
Liczba neuronów to podstawowy element wpływający na rozmiar sieci, jej zdolności do uczenia się. Ponieważ liczby neuronów na wejściu i wyjściu nie mogą być modyfikowane, badamy tylko zaleznośc sprawności od liczby neuronów w warstwie ukrytej.

Badano wartości:
* n = 2
* n = 3
* n = 5
* n = 10
* n = 20
* n = 40
* n = 75
* n = 100
* n = 150
* n = 200
* n = 250
* n = 300

#### Wnioski:
Liczba neuronów wydaje sięnie wskazywać na jakąkolwiek tendencję w skuteczności klasyfikacji. Możliwe że problem jest zbyt prosty lub zbyt złożony aby zadane liczby neuronów były w stanie go dobrze odwzorować. Dlatego wybrano domyślną wartość liczby neuronów, tj. średnią z liczby neuronów na wejściu i na wyjściu - czyli w tym wypadku n = 5.

Szczegółowe testy znajdują się w pliku *Wybor_liczby_neuronow.ipnb*.

## Testy modelu
Gotowy model z dobranymi parametrami przetestowano szczegółowo w pliku *Testy_modelu.ipnb*. Znajdują się tam także wykresy charakterystyki ROC, które wskazują, że model zwykle bardzo "pewnie" przewiduje wyniki. Prawdopodobnie wynika to z charakterystyki zbioru danych, lub samego problemu który może nie być odpowiedni dla machine learningu.

Dużą zmianę działania modelu może wnieść przesunięcie progu klasyfikacji transakcji jako fałszywej, w wyżej wymienionym pliku przedstawiono testy dla przykładowych wartości. 

Trodno ocenić wyniki tych testów, bo próg powinien w duzym stopniu być dopasowany do celów projektu, ale przyjęto wartość n = 0.1 jako optymalną, ponieważ w tym przypadku istnieje taka sama szansa błędu false-positive co false-negative (około 1 do 14).





