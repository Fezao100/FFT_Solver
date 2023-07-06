import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.pyplot import figure

#FUNÇÃO QUADRADA
n_amostras = 1024
t = np.linspace(0, 1, 1024, endpoint=False) #Pulsos por segundo
y=np.zeros(n_amostras) #Cria matriz 

y = signal.square(4 * np.pi * 5 * t) #Período da função
print(y)
np.savetxt('sinaldeentrada.txt',y)

#FFT E PLOTAGEM DOS GRÁFICOS
np.set_printoptions(threshold=8)

sinalentrada = np.loadtxt('sinaldeentrada.txt')

def fft(sinalentrada):
    sinalentrada = np.asarray(sinalentrada, dtype=float)
    N = len(sinalentrada)
    if N % 2 > 0:
        if N == 1 :
            return sinalentrada
    
        raise ValueError("tamanho da entrada tem que ser uma potencia de 2")
        
    else:
        Xpar = fft(sinalentrada[::2])
        Ximp = fft(sinalentrada[1::2])
        fator = np.exp(-2j*np.pi*np.arange(N)/N)
        sinal_saida = np.concatenate([Xpar + fator[:int(N/2)] *Ximp,Xpar + fator[int(N/2):] *Ximp])
        return sinal_saida

print("MOSTRANDO SINAL DE ENTRADA: ")
print( fft(sinalentrada))

#Função para criar eixo x do gráfico no domínio da frequência
def fft_freqs(sinalentrada,freq_aq):

    freqFft = np.fft.fftfreq(len(sinalentrada),d=1/freq_aq)
    #freqFft =freqFft[:int(len(sinalentrada)/2)] # Utiliza somente frequências abaixo da frequência de Nyquist
    return freqFft

#Lógica do programa
sinalentrada = np.loadtxt('sinaldeentrada.txt') #Carrega o sinal de entrada do arquivo txt
freq_aq = 1024 #Frequência na qual o sinal de entrada foi adquirido
x_tempo = np.linspace(-10,10,len(sinalentrada)) #Cria eixo x do gráfico no tempo
x_tempo2 = np.linspace(-1,1,len(sinalentrada))
saida_fft = fft(sinalentrada) #Calcula fft do sinal

def pegarfase(saida_fft):
    fase = np.angle(saida_fft)
    return fase

fase = pegarfase(saida_fft)

print(len(sinalentrada))

frequencias = fft_freqs(sinalentrada, freq_aq) # Cria eixo x
print(frequencias)
#frequencias = np.linspace(0,200,1)

#Tratando saídas
saida_fft_1 = (np.abs(saida_fft)/len(sinalentrada))*2
#saida_fft_1 = saida_fft_1[:int(len(sinalentrada)/2)]

#Plotando os gráficos
f, plots = plt.subplots(2,2,figsize=(20,10))
plots[0,0].plot(x_tempo, sinalentrada, color='#0f0')
plots[0,0].set_title('Sinal de Entrada')
plots[0,0].set_xlim([-10,10])
plots[0,0].grid()
plots[0,1].plot(frequencias, fase, color='#0f0')
plots[0,1].set_title('Fase')
plots[0,1].set_xlim([-90,90])
plots[0,1].grid()
plots[1,0].plot(saida_fft, color='#00f')
plots[1,0].set_title('Saida FFT')
plots[1,0].grid()
plots[1,1].plot(frequencias,saida_fft_1, color='#00f')
plots[1,1].set_xlim([-90,90])
plots[1,1].set_title('Saida FFT Após Trato')
plots[1,1].grid()

plt.subplots_adjust(hspace=0.8)
f.tight_layout(pad=3.0)
np.savetxt('saidafft1.txt',saida_fft_1)
plt.show()