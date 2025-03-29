# Algorytmy ML w cpp

## Plan
- [ ] Klasa `Dataframe` do wczytywania danych z pliku `.csv` 
    - metoda `shuffle` do przetasowywania danych w losowy sposób (za pomocą algorytmy [Fishcer-Yates](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)) 
    - metoda `train_test_split` - do podzielenia danych na zbiór testowy i treningowy,
    - Myślę że dane będziemy przechowywać jako [vector](https://en.cppreference.com/w/cpp/container/vector) [shared_pointer-ów](https://www.youtube.com/watch?v=4bdp9aHzuQY) do tablicy, gdzie tablice będą naszymi wierszami.
      - dzięki temu gdy będziemy tworzyć now Dataframe-y to zamiast kopiować wszystkich danych będziemy tylko kopiować wskaźniki do nich.
      - `shared_pointer` bardzo ułatwi życie w zarządzaniu pamięcią. Bedziemy korzystać z [dynamicznie alokowanych tablic](https://mattomatti.com/pl/cp14), bo niewiemy ile kolumn będzie w pliku csv
- [x] Base class `Model` dla naszych algorytmów ML
  - Kiedy będziemy implementować poszczególne modele to z tej klasy będziemy [dziedziczyć](https://www.youtube.com/watch?v=ZesZXlBcROA).
    Potem się to nam przyda (by potem wrzucać modele do tych samych funkcji dzięki [poliformizmowi](https://www.youtube.com/watch?v=9hGPe6BnTY4))
  - zaimplementować [wirtualne](https://www.geeksforgeeks.org/virtual-function-cpp/) metody: `fit`, `predict`
- [ ] Model - [KNN](https://www.youtube.com/watch?v=HVXime0nQeI)
- [ ] [Walidacja krzyżowa](https://pl.wikipedia.org/wiki/Sprawdzian_krzy%C5%BCowy)
- [ ] funckja do sprawdzania [Dokładności precyzji i czułości](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=pl)
- [ ] Model - [Regresja liniowa](https://www.youtube.com/watch?v=7ArmBVF2dCs)
- [ ] Model - [Regresja logistyczna](https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe)
- [ ] Model - [Drzewa decyzyjne](https://www.youtube.com/watch?v=_L39rN6gz7Y)
- [ ] *To be continued...*