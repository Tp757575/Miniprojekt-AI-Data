import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Indlæser datasættet
df = pd.read_csv(r"C:\Users\thoma\Desktop\python_work\Mini_projects\AI&DATA Miniprojekt\DailyDelhiClimateTrain.csv")

# Konvertere 'date' til datetime og brug som indeks
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Udfylder manglende værdier i 'meantemp' med seneste kendte værdi
df['meantemp'] = df['meantemp'].ffill()

# Plotter den originale tidsserie
plt.figure(figsize=(12, 6))
plt.plot(df['meantemp'], label='Original Meantemp Data')
plt.title('Original Meantemp Data')
plt.xlabel('Dato')
plt.ylabel('Meantemp')
plt.legend()
plt.grid(True)
plt.show()

# Angiver vinduestørrelse til medianfilter
window_size = 30

# Anvender glidende medianfilter
df['meantemp_med'] = df['meantemp'].rolling(window=window_size, center=True).median()

# Plotter original data og medianfiltreret data
plt.figure(figsize=(12, 6))
plt.plot(df['meantemp'], label='Original Data', alpha=0.5)
plt.plot(df['meantemp_med'], label=f'{window_size}-dages Medianfilter', linewidth=2)
plt.title('Meantemp Data – Medianfilter')
plt.xlabel('Dato')
plt.ylabel('Meantemp')
plt.legend()
plt.grid(True)
plt.show()

# Udføre FFT og filtrer høje frekvenser
meantemp_fft = fft(df['meantemp'])
n = len(df['meantemp'])
frequencies = np.fft.fftfreq(n, d=1)
cutoff = 0.1
meantemp_fft_filtered = meantemp_fft.copy()
meantemp_fft_filtered[np.abs(frequencies) > cutoff] = 0

# Gendaner tidsserien fra FFT (invers)
meantemp_filtered = ifft(meantemp_fft_filtered)
df['meantemp_fft_filtered'] = np.real(meantemp_filtered)

# Plotter original og FFT-filtreret data
plt.figure(figsize=(12, 6))
plt.plot(df['meantemp'], label='Original Data', alpha=0.5)
plt.plot(df['meantemp_fft_filtered'], label='FFT-filtreret Data', linewidth=2)
plt.title('Meantemp Data – Frekvensfilter')
plt.xlabel('Dato')
plt.ylabel('Meantemp')
plt.legend()
plt.grid(True)
plt.show()
