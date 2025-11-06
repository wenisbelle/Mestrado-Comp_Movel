Comparação: diferentes parâmetros, por exemplo de máscara e etc. E de fitness= distancia do drone, numero de membros do cluster. Comparar com o modelo aleatório e com esse híbrido que vou testar. 


Surveillance: Implementa o modelo que dá o valor de 1 quando a camera passa e vai diminuindo o valor da confiança

Surveillance_v2: Implementa o valor incremental de incerteza, sendo 0 quando a camera passa e esse valor cresce linearmente com o tempo.

Surveillance_v3: Teste para poder passar os valores via a script de execution.py



FALTA FAZER
Falta implementar um modelo que faça um ir pra uma direção e outro pra outra.


Trabalho para o próximos artigos: 
Não normalizar o map. Fazer com que 0 seja logo após visitar e esse valor vá subindo. Nesse caso, as regiões mais incertas serão as de maior valor de célula. Aí criar um fitness que possa se basear em algo tipo assim: A distributed task allocation approach for multi-UAV persistent monitoring in dynamic environments. 

Tunar a função de fitness por meio de algoritmos genéticos: tunar os parametros da equação do fitness e da máscara. 

No futuro, implementar kmeans para separar os clusters das regiões. 


