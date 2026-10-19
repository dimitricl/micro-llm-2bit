# Pilote QA #2 — échantillon qualitatif (~20 générations brutes)

Décodage de déploiement : greedy, repetition_penalty=1.15,
no_repeat_ngram_size=3, max_new=32. Comparaison directe pilote #1 vs #2
sur les mêmes prompts (extraits de samples.md).

## [capitales/par_coeur] cap-01-pc

prompt  : La France est un pays dont la capitale est
attendu : Paris
#1 gen  : ' Paris. La capitale de la France porte le nom de Paris. Sur le plan administratif, la capitaine de la fonction de la'  HIT
#2 gen  : ' Paris. On appelle Paris la capitaine de la France. La capitaine est un peu plus tard. La France a pour capitale Paris.'  HIT

## [capitales/par_coeur] cap-02-pc

prompt  : La capitale de l'Italie est
attendu : Rome
#1 gen  : " Rome. La capitale est Rome, c'est Rome. Le chef-lieu de l’Italien est Rome (1860-"  HIT
#2 gen  : " Rome. Rome est la capitale du Portugal. La capitale est Rome, c'est Rome. La ville de Rome est un pays dont la capit"  HIT

## [capitales/par_coeur] cap-03-pc

prompt  : L'Espagne est un pays dont la capitale est
attendu : Madrid
#1 gen  : " Madrid. La ville de Madrid est la capitaine de l'Escher. L'Esquier : capitale Madrid. Le chef-lie"  HIT
#2 gen  : " Lille. On appelle Lille la capitaine de l'Escrive. La capitaine est un premier pays dont le nombre de pays"  miss

## [capitales/reformule] cap-01-rf

prompt  : Quelle ville est la capitale de la France ?
attendu : Paris
#1 gen  : '\nLa France est un pays dont la capitaine est Paris. On appelle Paris la capitelle de la fête de la république'  HIT
#2 gen  : '\nLa France est un pays dont la capitaine est une région de la régime où le peuple étaient déjà é'  miss

## [capitales/reformule] cap-02-rf

prompt  : Quelle est la capitale de l'Italie ?
attendu : Rome
#1 gen  : '\nLe chef-lieu de l’Italien est Rome. La ville de Rome est la ville.\nLe capitale du Rome est'  HIT
#2 gen  : "\nLe , la capitaine de l’Italien, c'est la capitelle de l.\nLe chef-lieu de l'sis"  miss

## [dates/par_coeur] dat-01-pc

prompt  : la Révolution française date de
attendu : 1789
#1 gen  : " 1960. La capitale de l'Allemagne est une année de 365 jours. La révélation"  miss
#2 gen  : ' 1799. La République françoise a eu lieu en 1800. La républéque f'  miss

## [dates/par_coeur] dat-02-pc

prompt  : la prise de la Bastille a eu lieu en
attendu : 1789
#1 gen  : ' 1963. La capitale de la ville de Montréal est la capitale du Canada. Le Canada : capitale Ottawa. Le'  miss
#2 gen  : ' 1792. la prise du roi de France date de 1800. La prise des rois de France a eu'  miss

## [dates/par_coeur] dat-03-pc

prompt  : le sacre de Napoléon date de
attendu : 1804
#1 gen  : ' 1464.\nLe chef-lieu de la France est un pays dont la capitale est Paris.\nLa ville de Paris est'  miss
#2 gen  : " 1789. La bataille de Napole a eu lieu en 1800. La Bataille d'Auver"  miss

## [dates/reformule] dat-01-rf

prompt  : Quand a eu lieu la Révolution française ?
attendu : 1789
#1 gen  : "\nLa révélation de l'eau est de 1960 à 1890.\nLa lumière se"  miss
#2 gen  : "\n1919 : La République françoise, c'est une républéque.\n1890 :"  miss

## [dates/reformule] dat-02-rf

prompt  : En quelle année a eu lieu la prise de la Bastille ?
attendu : 1789
#1 gen  : "\nLa capitale de l'Argentine est une capitale. La capitale est un pays dont la capitale se trouve à l'ou"  miss
#2 gen  : '\n1994 : Le début de la guerre de Trente Ans : 1989.\n1000 : Le premier'  miss

## [definitions/par_coeur] def-01-pc

prompt  : Le mot « photosynthèse » désigne
attendu : chlorophylle, lumière
#1 gen  : " le symbole Au. Le symbole chimique de l'or est Au. L'or a pour symbole de l’or est noté Au"  miss
#2 gen  : " une valeur de référence.\nLe mot de la vie\nLe nombre d'arêtes d'un corps est de"  miss

## [definitions/par_coeur] def-02-pc

prompt  : la gravité est
attendu : attraction, masse
#1 gen  : ' de 365 jours. La Terre met 3000 jours pour faire le tour du Soleil. La péri'  miss
#2 gen  : " la formule de l'équation de la Terre. La gravité de l’équations est la fonction de l‘é"  miss

## [definitions/par_coeur] def-03-pc

prompt  : Le mot « monarchie » désigne
attendu : roi
#1 gen  : ' le nom de « révolution de la liberté ».\nLe terme « résultat » est un pays dont la capitale est'  miss
#2 gen  : " un établissement qui se trouve à l'époque. la monarchie : un événement qui est un éto"  miss

## [definitions/reformule] def-01-rf

prompt  : Comment définit-on la photosynthèse ?
attendu : chlorophylle, lumière
#1 gen  : "\n\nLe symbole chimique de l'or est Au.\nLe système de l’or est noté Au. L'or a"  miss
#2 gen  : '\n\n\n'  miss

## [definitions/reformule] def-02-rf

prompt  : Que signifie « gravité » ?
attendu : attraction, masse
#1 gen  : "\nLe point d'é"  miss
#2 gen  : "\nLe mot « gravitée », c'est un mot « général ». Dans la générale, il y a un"  miss

## [unites_sciences/par_coeur] uni-01-pc

prompt  : L'eau bout à
attendu : 100
#1 gen  : ' 100 degrés'  HIT
#2 gen  : " 1000 mètres par seconde. La température d'é"  miss

## [unites_sciences/par_coeur] uni-02-pc

prompt  : L'eau gèle à
attendu : 0
#1 gen  : ' 0 degré'  HIT
#2 gen  : " la mer. Le mot « climat », c'est un climat qui caractérise la commune."  miss

## [unites_sciences/par_coeur] uni-03-pc

prompt  : Un kilomètre vaut
attendu : 1000
#1 gen  : ' 1000 m. Un kilométre vous avez 1,80 mètres. Un kilogramme vaut mil'  HIT
#2 gen  : ' 1000 mètres. Un kilométre vous avez 1,001 mètre. Un kilogramme'  HIT

## [unites_sciences/reformule] uni-01-rf

prompt  : À quelle température l'eau bout-elle ?
attendu : 100
#1 gen  : "\nLa températures d'é"  miss
#2 gen  : "\nAu début de la Seconde Guerre mondiale, la ville de Paris est la capitale de l'Espagne"  miss

## [unites_sciences/reformule] uni-02-rf

prompt  : À quelle température l'eau gèle-t-elle ?
attendu : 0
#1 gen  : "\nLa températures de congélation de l'élément or est de 100 degrés"  miss
#2 gen  : '\n\n\nLa températures de la températuré :\nLa vitesse de la lumière est de 100 degr'  miss

