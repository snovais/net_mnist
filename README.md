# dense_net_mnist

Redes Neurais Artificiais para Classificação de Imagens de Dígitos

Redes Neurais Artificiais consistem em modelos matemáticos que simulam as características dos neurônios
cerebrais. Diante disso é possível utiliza-las para classificação, predição, reconhecimento de objetos, 
síntese de imagens e entre outras possibilidades. Este projeto consiste em um modelo criado utilizando a
linguagem de programação Python e o Framework do Google, Tensorflow em sua versão 2.x. A Rede Neural Artificial
Densa caracteriza-se por uma camada de neurnônios na entrada com 728 unidades utilizando a função de 
ativação RELU (Unidade Linear Retificada), após a camada de entrada encontram-se outras 5 camadas ocultas
contendo em cada uma 1024 neurônios. Os pesos são gerados automaticamente pelo Tensorflow. Entre as camadas
ocultas foram adicionadas camadas de Dropout, este serve para eliminar aleatoriamente e temporariamente 
alguns dos neurônios ocultos na rede, deixando os neurônios de entrada e saída intocados. Isto funciona 
como se estivesse treinando redes neurais diferentes. E assim, o procedimento de eliminação é como calcular 
a média dos efeitos de um grande número de redes diferentes. As diferentes redes se adaptarão de diferentes
maneiras, e assim, esperançosamente, o efeito líquido do Dropout será reduzir o sobre ajuste. Depois dessas 
camadas, encontra-se a camada de saída que possui dez neurônios para indicar a classificação de dígitos variando
de 0 a 9. No entanto, porque desenvolver mais um projeto dentro tantos que já existem? 
A resposta é simples, pois o tensorflow tem sido constantemente atualizado e a muitos exemplos que a comunidade
produz não está em português brasileiro e ainda a os Desenvolvedores do Tensorflow aplicaram novas melhorias 
como Fita Gradiente, em que é utilizado como uma espécie de memória para a rede lembrar das posições dos pesos
e assim ao recordar para onde deve ir na retropropagação, então é possível obter melhorias de perfomance. 
Além disso, foi implementado um decorator "@tf.function", este cria grafos para funções e isso permite acelerar 
mais o processamento.



Referências

https://www.tensorflow.org/guide
http://deeplearningbook.com.br/capitulo-23-como-funciona-o-dropout/
