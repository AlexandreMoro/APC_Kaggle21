# APC_Kaggle21
# Pràctica Kaggle APC UAB 2021-22

### Nom: ALEXANDRE MORO I RIALP
### DATASET: Melbourne Housing Market
### URL: [kaggle](https://www.kaggle.com/anthonypino/melbourne-housing-market)


## Resum

El dataset de Kaggle en el què es centra aquest treball conté dades dels habitatges venuts a la ciutat de Melbourne, Austràlia, entre els anys 2016 i 2018. 
El Kaggle ofereix dos datasets, un simplificat amb pocs atributs i un amb més informació (característiques) per cada habitatge venut. 
S'ha optat per treballar amb aquest segon al contenir major informació. 
Aquest dataset s'anomena Melbourne_housing_FULL.csv. Conté un total de 34857 mostres amb 21 atributs:'Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method',
 'SellerG', 'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude',
 'Regionname', 'Propertycount'). Un 38%, d'ells és categòric i la resta són numèrics de tipus float64. Aquests darrers valors no es troben normalitzats i tenen unes grans diferènices d'escala ja que per exemple, els valors del nombre d'habitacions o banys són de poques unitats mentre que el preu supera en moltes ocasions el milió.

Les mostres presenten una gran heterogeneïtat pel què fa a les característiques que tenen registrades (sense valors null). En alguns atributs com BuildingArea o YearBuilt, 66% de les 
mostres tenen Nan en aquestes caracterísitques. Per la resta d'atributs també hi ha una gran presencia de valors null. Només un 25% de les mostres contenen tota la informació als atributs. 



### Objectius del dataset

L'objectiu del treball amb aquest dataset són dos. Per una banda, aprendre a gestionar i resoldre les mostres incompletes per poder aprofitar la major quantitat de mostres possibles. 
La segona, crear models regressors que permetin predir  el preus  de venda per als habitatges que es posaran al mercat en el futur a la ciutat de Melbourne.

## Experiments

Durant el desenvolupament d'aquesta pràctica s'han realitzat diferentes tasques i experiments. En un primer moment s'ha analitzat les dades del dataset. Posteriorment, s'ha 
procedit a tractar els valors Nan degut a la seva gran presència en el dataset. Un cop preparades les dades, s'ha porcedit a creaer models regressors. Concreatament s'han generat 
3 tipus de regressors: el regressor Lineal, el regressor Random Forest i el regressor Decision Tree. Per cadasqun d'aquest tipus de model, s'ha aplicat diferents K.fold d'entrenamen  així com l'efecte, sobre les prediccions generades, de la variació del paràmetre max_depth pels dos darrers. Amb el millor dels  dels models, Random Forest Reg., s'ha cercat l'optimització dels seus hiper-paràmetres per mitjà de l'algorisme de cerca RandomSearchCV. Finalment, s'ha repetit l'entrenament sobre el mateix dataset però realitzant un 
pre-processat simple consistent en eliminar totes les mostres amb algun valor null en alguna de les seves característiques. 


### Preprocessat

Després de la fase d'exploració del dataset s'ha identificat la gran quantitat de valors Nan en les mostres com a un factor que podia empobrir els resultats obtinguts. Per solucionar-ho,
s'ha aplicat un processament del dataset amb la intenció de netejar, enriquir i integrar les dades i poder realitzar la creació de models amb una major quantitat d'informació.

El dataset conté, per cada mostra, un conjunt de 21 aributs que aporten informació descriptiva sobre l'habitatge. Entre aquests, s'ha  identifificat 7 que aportaven informació redundant ja que d'una forma o altra descrivien la localització dels immobles. Es tracta dels atributs: 'Suburb','Address','Postcode','CouncilArea','Latitude','Longitude', 'Regionname'. D'entre aquests, s'ha considerat que Lattitude i Longtitude eren els atributs que més convenienment donaven la localització ja que són numèrics i a diferència de l'altre numèric, Postcode, són únics (a excepció dels pisos). Així doncs, s'ha optat per eliminar els 5 atributs amb informació redundant. 

A causa de la important presència de Nan en les mostres ha calgut idear una forma de substituir-los sense afegir informació errònia. Per als atributs Lattitude, Longtitude, YaerBuilt i Landsize s'ha considerat adequat substituir els valors null de les mostres per la mitjana de totes les mostres pròximes en la seva localització. Per fer-ho s'ha optat per agrupar les mostres segons l'atribut Suburb (abans d'eliminar-lo) ja que és el que tenia un major nombre de valors únics, és a dir, era més específic en identificar una zona. 

Pel cas dels atributs Price, Bathroom, Car i BuildingArea, s'ha optat per eliminar les mostres amb nulls ja que pel primer, Price, aquest és el valor objectiu de les prediccions i susbtituir-lo segons algun criteri hauria estat hardcodejar la tasca dels models regressors, i pels altres tres, no s'ha identificat un patró que assegurés que la substitució no aportés error al dataset.

Per als atribut Method, Type i SellerG que identifiquen el mètode de venda, el tipus d'habitatge i el venedor, al ser atributs categòrics, s'ha optat per convertir en nous atributs els seus possibles valors de tal forma que s'han creat 8 columnes que descrivien de forma bolean l'informació.

L'atribut Date ha estat transformat a númeric seleccionant l'any de venda ja que al ser Price l'atribut objectiu, aquest es veurà afectat per les diferències de valor de la moneda entre cada any. 

Precisament, per compensar les variacions del valor econòmic de la moneda en què es feien la venda i que determinava el valor del preu, s'ha optat per equiparar el valors de l'atribut Price per cada any tenint en compte la inflació que el dòlar australià va experimentar entre els anys 2016, 2017 i 2018. Així doncs, s'ha actualitzat el valor de Price per cada mostra en la relació següent: {'2016': 1.035, '2017': 1.019, '2018': 1.0}.

Després d'aplicar aquest preprocessament, el dataset ha quedat amb 10469 mostres i 22 atributs sense cap valor a null. Per contra un preprocessament simple basat en eliminar totes les mostres amb valors Nan i suprimir els atributs categorics, hauria reduït el dataset a 8887 mostres i 13 característiques. Per tant, el preprocessat realitzat ha permés utilitzar un dataset amb un 20% més de mostres i més informació.



### Model

Recull dels models creats per cada dataset amb el millors paràmetres i resultats:

| Model            | Dataset | Hiperparametres | Mètrica   | Temps |
|------------------|---------|-----------------|-----------|-------|
| Regressor Lineal | EDA     | default         | R2: 0.57  | 6 ms  |
| Regressor Lineal | FULL    | default         | R2: 0.6   | 3 ms  |
| Random Forest    | EDA     | max_depth: 35   | R2: 0.809 | 3.42s |
| Random Forest    | FULL    | max_depth: 30   | R2: 0.774 | 2.25s |
| Decision Tree    | EDA     | max_depth: 7    | R2: 0.682 | 29ms  |	
| Decision Tree    | FULL    | max_depth: 8    | R2: 0.717 | 22ms  |




## Demo
Per tal de fer una prova dels resultats pels models entrenats sobre EDA, s'ha extret del dataset EDA un conjunt de mostres (100) abans de l'entrenament, es pot fer servir la següent comanda:  python3 demo/demo.py


## Conclusions

Entre els diferents models generats en aquest treball, els models de Random Forest Regressor han donat les millors prediccions. El resultats del coeficient de determinació R2 d'aquests models han estat força consistents entorn al 0.8. Pel què fa als models de tipus Decision Tree regressor, ho han estat entorn al 0.7. Finalment, els pitjors resultats els han donat el Regressor Lineal amb entorn a 0.6 d'R2 score. L'avantatge del Random Forest radica en l'ús de diferents subsets de les mostres per entrenar diferents arbres de decisió i  utilitzar la mitjana de les seves prediccions. D'aquesta forma aconsegueix millorar els resultats de la predicció i controlar l'overfiting respecte a un únic arbre sobre tot el dataset. La contrapartida d'aquest model és que al fer un ensemble, el seu cost computacional és molt superiror i fer els experiments amb ell requereix força temps.

Comparant els resultats dels models entre els entrenats sobre el dataset sense substitució de Nan, Full, i el pre-processat amb substitució, EDA, podem observar com l'increment aconseguit d'un 20% de les mostres no aporta una millora en les prediccions dels models ja que s'obtenen resultats semblants. D'aquest fet podem extreure'n dues conclusions. La primera és que el volum de mostres de 8800 del Full és suficientment representativa de tots els possibles habitatges que es poden vendre a la ciutat i augmentar la quantitat de mostres només aporta solapament de mostres. La segona, és que un dels perills de substituir valors desconeguts és generar informació errònia que faci equivocar els models en el seu entrenament. Al haver aconseguit resultats semblants, podem considerar que els substitucions van ser correctes i no van inserir valors errònis en les mostres tractades.

La major part dels treballs realitzats amb aquest Kaggle es centren en la part de tractament de Nan i l'anàlisi de les dades. Pels que van un pas més enllà i generen models de predicció, la majoria opta per treballar amb el dataset simplificat i aconsegueixen R2 escort d'entre 0.4 i el 0.7. El treball amb el dataset més extens i l'ús del Random Forest Regressor ha permés aconseguir uns resultats millors. 



## Idees per treballar en un futur

De cara a una extensió d'aqeust treball seria interessant aplicar algun algorisme diferent com el KNeighborsRegressor o algunds dels mètodes d'ensemble més complexes com el Voting o Boosting en diferents formes. Aquests tenen un cost computacional major i requereixen més temps. Per altra banda seria adequat intentar descubrir els patrons que permetessin susbtituir els valors Nan de les característiques que s'han hagut d'eliminar per no introduir informació equivocada com BuildingAre o  Car.

## Llicencia
El projecte s’ha desenvolupat sota llicència oberta Apache 2.0.





