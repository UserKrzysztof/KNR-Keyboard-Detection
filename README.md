# KNR-Keyboard-Detection

## Mapa
Poniżej mapa jak znaleźć najważniejsze pliki w repo
```python
KNR-Keyboard-Detection
│   README.md
│
└───data_generation
│   │   only_keyboard.blend # środowisko go generowania klawiatury i maski na całą klawiaturę
│   │   keys_and_keyboard.blend # środowisko do generowania klawiatury i maski z wyróżnieniem przycisków na klawiaturze
│   │
│   └───random_textures # randomowe tekstury na otoczenie klawiatury
│   │    │   gray_0.png
│   │    │   gray_1.png
│   │    │   ...
│   │  
│   └───data #puste foldery na wygenerowane dane (rozmiar danych był za duży na github, trzeba pomyśleć jak je sobie udostępniać)
│ 
└───code
    │   random_textures.ipynb #generowanie randomowych tekstur
    │   mask_operation.ipynb #pliki w którcyh próbowałem znaleźć przyciski na podstawie maski na całą klawiature
    │   finding_keys.ipynb #pliki w którcyh próbowałem znaleźć przyciski na podstawie maski na całą klawiature
    │   just_unet.ipynb #najnowszy pomysł rozwiązania, niestety zbyt potężny na mój komputer
    │   pytorch_knr.yml # są w nim wypisane paczki
```

## Jak wygenerować dane?
1. Uruchom jeden z plików ```.blend```
2. W zakładce Scripting powinien się pojawić skryt ```generating_data_cope.py```
3. Zmień wartość zmiennej ```WORK_DIR``` na ścieżkę do foledru objemującą plik ```.blend```
4. Przy pierwszym uruchomieniu skryptu w danej sesji Blendera odkomentuj linię 220: ```update_file_paths()```. Przy następnych wywołaniach powinna być zakomentowana.
5. W zakłądce Compositing i nodzie File Output zmień ścieżkę w Base Path. W tym folderze zapiszą się wygenerowane dane

> [!NOTE]
> Jedno zdjęcie może potrzebować ok. 10 s na wyrenderowanie. Użyty silnik dość mocno obciąża grafikę.
>
