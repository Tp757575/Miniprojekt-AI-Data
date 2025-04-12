import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Indlæs datasættet fra CSV-filen
df = pd.read_csv(r"C:\Users\thoma\Desktop\python_work\Mini_projects\AI&DATA Miniprojekt\DailyDelhiClimateTrain.csv")

# Konverter 'date'-kolonnen til datetime-format og sæt den som indeks
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Fyld manglende værdier ud i 'meantemp' med forrige værdier
df['meantemp'] = df['meantemp'].ffill()

# Visualisering af den originale tidsserie
plt.figure(figsize=(12, 6))
plt.plot(df['meantemp'], label='Original Meantemp Data')
plt.title('Original Meantemp Data')
plt.xlabel('Dato')
plt.ylabel('Meantemp')
plt.legend()
plt.show()

# Definer vinduesstørrelsen - 30 dage
window_size = 30

# Beregn medianfilteret (glidende median)
df['meantemp_med'] = df['meantemp'].rolling(window=window_size, center=True).median()

# Plot original data og den medianfiltrerede data
plt.figure(figsize=(12, 6))
plt.plot(df['meantemp'], label='Original Data', alpha=0.5)
plt.plot(df['meantemp_med'], label=f'{window_size}-dages Medianfilter', linewidth=2)
plt.title('Meantemp Data - Medianfilter Præprocessering')
plt.xlabel('Dato')
plt.ylabel('Meantemp')
plt.legend()
plt.show()


# Eksempelvis bevarer vi stadig den FFT-filtrering, hvis du ønsker at sammenligne
meantemp_fft = fft(df['meantemp'])
n = len(df['meantemp'])
frequencies = np.fft.fftfreq(n, d=1)
cutoff = 0.1
meantemp_fft_filtered = meantemp_fft.copy()
meantemp_fft_filtered[np.abs(frequencies) > cutoff] = 0
meantemp_filtered = ifft(meantemp_fft_filtered)
df['meantemp_fft_filtered'] = np.real(meantemp_filtered)

plt.figure(figsize=(12, 6))
plt.plot(df['meantemp'], label='Original Data', alpha=0.5)
plt.plot(df['meantemp_fft_filtered'], label='FFT-filtreret Data', linewidth=2)
plt.title('Meantemp Data - Frekvensdomænefiltrering')
plt.xlabel('Dato')
plt.ylabel('Meantemp')
plt.legend()
plt.show()
