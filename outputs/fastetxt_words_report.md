## Interpretacja wyników FastText i t-SNE

---

### 1. Czy pretrenowane embeddingi dobrze odwzorowują znaczenie słów w korpusie?

**Częściowo tak.**
Model bardzo dobrze grupuje słowa tematycznie i semantycznie.

- **Sport:** `wisła` → `cracovia`, `kraków`, `lechia`, `łks`
- **Sport (kontekst):** `mecz` → `sparingu`, `pojedynku`, `dwumeczu`
- **Polityka:** `pis` → `lpr`, `sld`, `lepper`, `kaczynski`
- **Relacje:** `grać` → `zagrać`, `grywać`, `pogrywać`
- **Koncepty:** `mogę` → `muszę`, `potrafię`, `chcę`

**Ale:** model nie zna specyfiki korpusu i traktuje błędy oraz "śmieciowe" tokeny jak prawdziwe słowa. Na przykład token `nhttps` (prawdopodobnie z błędnie sparsowanego linku) jest uznawany za podobny do `0adminhttp`, `adminhttp` i `aspxhttp`.

**Wniosek:**
Model kgr10.skipgram dobrze odwzorowuje ogólne znaczenia leksykalne i kategorie, ale gorzej radzi sobie z językiem nieformalnym, błędami i szumem informacyjnym (noise) obecnym w Twoim korpusie.

---

### 2. Czy w korpusie widać grupowanie słów związanych z anotacją?

Tak — widać bardzo wyraźne tendencje.

| Etykieta                 | Typowe słowa                                 | Charakterystyka                                                                                                                                                                                                                    |
| :----------------------- | :------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **contains harmful**     | chuj, popierdoliło, pis, polaków, pisu       | Wulgaryzmy, mocne nacechowanie emocjonalne, tematy polityczne i narodowościowe, które często wywołują skrajne opinie.                                                                                                              |
| **normal**               | wiem, sie, mogę, wisła, meczu, masz          | Neutralne, popularne słowa konwersacyjne oraz powszechne tematy (np. sport). _Uwaga: Słowo `kurwa` też tu występuje, co sugeruje, że ta lista to "najczęstsze słowa w dokumentach 'normal'", a nie "słowa unikalne dla 'normal'"_. |
| **contains harmfulrmal** | zły, gnoju, spierdolił, macierewicz, studiów | Słowa nacechowane negatywnie (`zły`, `gnoju`), wulgarne (`spierdolił`) oraz specyficzne tematy (polityka, edukacja), które mogą być pejoratywne w kontekście.                                                                      |

Widać, że embeddingi grupują słowa zgodnie z ich tematem i tonem,
czyli pośrednio odwzorowują anotację. Nie są to jednak klastry czysto emocjonalne — raczej tematyczne (sport, polityka, wulgaryzmy).

---

### 3. Czy widać grupowanie na słowa pozytywne i negatywne?

**Częściowo.**
Model nie grupuje słów na podstawie prostego podziału "pozytywne / negatywne", ale na podstawie **kontekstu użycia**.

Najlepszym przykładem jest `lepiej`, które ma w pobliżu zarówno `fajniej` i `ładniej` (pozytywne), jak i `gorzej` (negatywne). Model rozumie, że są to słowa tej samej _kategorii_ (przysłówki porównawcze opisujące jakość), a nie, że mają ten sam _wydźwięk_.

Jednocześnie widać silne klastry słów negatywnych, np. `zły` → `niedobry`, `beznadziejny`, `kiepski`, `paskudny`. Dzieje się tak prawdopodobnie dlatego, że te słowa są używane w bardzo podobnych strukturach gramatycznych (np. "ten mecz był [zły/kiepski/beznadziejny]").

**Wniosek:**
FastText nie rozumie sentymentu (emocji). Rozumie kontekst gramatyczny i tematyczny. Aby rozróżnić emocjonalny ton, potrzebny byłby model kontekstowy (np. PolBERT lub HerBERT).

---

### 4. Jakie różnice mogą wystąpić między modelami i dlaczego?

| Model                                | Zalety                                                 | Wady                                                                    |
| :----------------------------------- | :----------------------------------------------------- | :---------------------------------------------------------------------- |
| **CBOW**                             | lepszy dla częstych słów, uśrednia znaczenie kontekstu | gorzej radzi sobie z rzadkimi słowami                                   |
| **Skip-gram**                        | lepiej dla rzadkich i specyficznych słów               | wolniejszy w treningu                                                   |
| **dim=100 vs dim=300**               | mniejsze embeddingi = szybsze, ale mniej precyzyjne    | większe embeddingi = bogatsze znaczenie, ale trudniejsze w wizualizacji |
| **Wikipedia vs Social Media corpus** | ogólna wiedza językowa                                 | brak slangu, emocji, ironii                                             |

Różnice w podobieństwach wynikają głównie z źródła danych treningowych –
model z Wikipedii będzie neutralny i formalny,
a model z Twittera odda emocje, ale będzie mniej „gramatyczny”.
