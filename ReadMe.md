# Descrição de cada um dos modelos

Surveillance: Implementa o modelo que dá o valor de 1 quando a camera passa e vai diminuindo o valor da confiança

Surveillance_v2: Implementa o valor incremental de incerteza, sendo 0 quando a camera passa e esse valor cresce linearmente com o tempo.

Surveillance_v3: Teste para poder passar os valores via a script de execution.py

Surveillance_recruting: Teste do sistema para efetivamente recrutar os outros UAVs. Nesse modelo, a câmera foi modificada para uma customCamera.py, capaz de pegar outros dados do nó. 

# TODO:

[x] Modificar a câmera para pegar os dados do nó
[x] Modificar os POIs para que possam se tornar ameaças, de forma aleatória, até um limite máximo de ameaças possíveis. E mudar a cor da visualização.
[x] UAVs caaptem a leitura quando é ameaça e salvam na variável: threats_found
[ ] Mudança de estado do UAV, de MAPPING para recruting
[ ] Implementar o algoritmo de recruting, que depende do level da ameaça
[ ] Implementar um algoritmo de engaggig
[ ] Implementar uma função no UAV capaz de mudar o estado do POI, quando as condições do engajamento forem satisfeitas. 


# TODO longo prazo:
[ ] Implementar energia
[ ] Implementar algoritmos de verificação de mensagem, para levar em conta problemas de comunicação e latência. 
[ ] TUnning por meio de GA
[ ] Falta implementar um modelo que faça um ir pra uma direção e outro pra outra.

