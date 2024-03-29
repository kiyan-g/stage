# stage 1A
## Introduction

Ce projet a pour objectif de découvrir les marchés financiers et plus précisément l’analyse technique qui y est faite. Mais qu’est-ce que l’analyse technique ? En finance on trouve deux types d’analyse : fondamentale et technique. Comme le souligne John Murphy dans son œuvre _L'analyse technique des marchés financiers_ :
>"Le fondamentaliste étudie les causes des mouvements de marché, 
>alors que le technicien étudie leur effets."

En effet, l’analyse fondamentale se préoccupe à la fois des publications économiques et de l’actualité ainsi que tous types de facteurs fondamentaux pour prévoir les mouvements à long terme sur le marché. Tandis que l’analyse technique ne se concentre que sur deux aspects, les prix et les volumes, afin d’étudier l’action des marchés. Son but étant d’essayer d’anticiper les futures tendances des prix à l’aide de graphiques notamment.

Lors de mon stage dans le pôle « Gestion quantitative » au sein de l’entreprise LBPAM, je vais : 
   - utiliser python 3.9
   - utiliser github
   - utiliser yahoo finance
   - construire des modèles informatiques qui permettent d’aider à la décision d’achat/vente 

## Outils

Pour mes codes j’aurai besoin d’utiliser plusieurs librairies python.
   - __Sickit-learn__ (0.24.2) : utilisé pour le machine learning et l'analyse prédictive
   - __Numpy__ (1.20.3) : utilisé pour manipuler les matrices et des fonctions mathématiques
   - __Pandas__ (1.2.4) : utilisé pour manipuler les dataframes
   - __Matplotlib__ (3.4.2) : utilisé pour tracer et visualiser des données sous formes de graphiques
   - __TA-lib__ (0.7.0) : utilisé pour effectuer des analyses techniques sur des données du marché financier

## Partie 1

L’objectif de cette partie est de réaliser un « backtesting » sur le bitcoin, càd tester notre stratégie et l’optimiser sur des anciennes données réelles pour ensuite l’utiliser sur des données futures. Apres avoir étudié et compris les différents indicateurs comme le RSI ou le EMA, j’ai construit des dataframes pour chaque indicateur en modifiant juste les périodes.

Ensuite je dois évaluer la qualité des indicateurs en fonction du cours que j’ai choisi, ici le bitcoin, afin de construire ma propre stratégie de marché. Pour cela je vais d’abord séparer mes données en « training » et « test » pour évaluer mes indicateurs sur des données passées. De plus, je construis une colonne ‘predictive price’ comportant des valeurs binaires sur chaque jour et indiquant la tendance (1 si ça monte, 0 sinon). Puis, à l’aide de ‘__sklearn__’ j’importe différents modèles de classification (comme _‘RandomForestClassifier’_) qui me permettront de classer les indicateurs du plus au moins fiable.

