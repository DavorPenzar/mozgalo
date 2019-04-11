# Biljeske i nejasnoće prilikom rješavanja problema sa Mozgala

*Upute za pisanje file-a*:
- srodnu skupinu bilješki/nejasnoća odvajati nizom znakova tipa '-'
- staviti oznaku **RIJEŠENO** na kraj skupine ukoliko smo tu skupinu riješili, te po potrebi napisati rješenje problema
- dijakritike, gramaticki i pravopisno ispravan tekst nisu obavezni

## Bilješke i nejasnoće:

- skuziti sto tocno oznacavaju stupci "Unnamed: 0.1" i "OZNAKA_PARTIJE"
- rijesiti pitanje stupca STAROST i njegovih nelogicnih vrijednosti - mozda izbaciti te podatke jer nema ih puno u odnosu na cijeli dataset
- rijesiti problem gdje je UGOVORENI_IZNOS 0(nista)
- vidjeti baratamo li samo sa kreditima ili i sa stednjama
- nadovezujuci se na problem iznad, kako je moguce da se preostali iznos kredita moze za jedan kvartal povecati za 77423644 (ukoliko baratamo samo sa kreditima)
- skuziti obuhvaca li nas dataset sve moguce kategorije iz stupaca "PROIZVOD", "VRSTA_PROIZVODA", "TIP_KAMATE"

**RIJESENO** --- > pogledati mail "Pitanja u vezi ovogodišnjeg zadatka"
--------------------------------------------------------------------------------------------------------------------

- pitati ih kako je moguce da i u trening i u test skupu podataka, koji je stavljen na CodaLab, postoje partije za koje datum otvaranja 
dolazi poslije planiranog datuma zatvaranja.Takvih partija u test skupu ima 2252. Smatramo da ovo isto pitanje, koje se odnosilo na trening skup, 
nije dobro objasnjeno u FAQ-u. Nase potencijalno objasnjenje je da sve partije koje imaju DO >= PDZ su zapravo produzenje ugovora koje se dogodilo nakon
PDZ i te su partije napravljene cisto iz neke bankarske formalnosti pa nije ni bitno koji im je novi PDZ
- moramo promijeniti vrijednosti stupca PRIJEVREMENI_RASKID na nacin koji su opisali u FAQ-u. NE ZABORAVITI
--------------------------------------------------------------------------------------------------------------------
