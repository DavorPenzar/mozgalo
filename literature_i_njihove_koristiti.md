http://arxiv.org/abs/0711.4452v1
---> ovo nam moze **dosta** pomoci u dva slucaja:
	
	1. ukoliko smo "istrenirali" taj RS-PCA na kategorickim varijablama katg(1), katg(2), ..., katg(n) za neki n prirodan broj, 
		mozemo lagano projicirati vrijednosti tih navedenih kategorickih varijabli u potprostor kojeg je pronasao RS-PCA
		Ovo je super jer recimo za svaki red(feature vector) u test set-u vrijednosti kategoricke varijable ne moramo prikazivati preko onih binarnih 
		ili dummy vrijednosti nego ih mozemo prikazati preko vektora koji je nastao projiciranjem na taj potprostor

	2. vrijednosti d-ova i c-ova (definiranih u clanku) mozemo iskoristiti za razmisljanje o razlicitim modelima koje bi primjenjivali na feature 
	vectorima koji se po necemu (recimo nekoj kategorickoj varijabli) razlikuju. Naime, ako je d(a,b) dovoljno velik, za a i b vrijednosti neke iste
	kategoricke varijable, onda to znaci da se feature vector koji ima vrijednost a (u toj spomenutoj kategorickoj varijabli) dosta razlikuje u ostalim featurima
	od feature vectora koji ima vrijednost b. To na neki nacin znaci da mozemo napraviti razlicite modele na osnovu vrijednosti a i b u toj kategoriji.