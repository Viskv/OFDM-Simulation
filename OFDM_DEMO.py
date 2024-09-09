import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# QAM modulation and demodulation functions
def qam_modulate(bits, qam_order):
    M = int(np.log2(qam_order))  # Number of bits per symbol
    symbols = []
    for i in range(0, len(bits), M):
        b = bits[i:i+M]
        symbol = 0
        for j in range(len(b)):
            symbol += int(b[j]) * 2**(len(b)-1-j)
        symbols.append(qam_constellation[symbol])
    return np.array(symbols)

def qam_demodulate(symbols, qam_order):
    M = int(np.log2(qam_order))  # Number of bits per symbol
    bits = ''
    for symbol in symbols:
        distances = np.abs(symbol - qam_constellation)
        closest_symbol = np.argmin(distances)
        b = format(closest_symbol, '0' + str(M) + 'b')
        bits += b
    return bits

# String to binary conversion
def string_to_binary(s):
    return ''.join(format(ord(c), '08b') for c in s)

# Binary to string conversion
def binary_to_string(b):
    return ''.join(chr(int(b[i:i+8], 2)) for i in range(0, len(b), 8))

# Generate QAM constellation points
def generate_qam_constellation(qam_order):
    levels = int(np.sqrt(qam_order))
    constellation = []
    for i in range(levels):
        for j in range(levels):
            real = 2 * i - levels + 1
            imag = 2 * j - levels + 1
            constellation.append(complex(real, imag))
    return np.array(constellation) / np.sqrt(2)

# Calculate Bit Error Rate (BER)
def calculate_ber(transmitted_bits, received_bits):
    bit_errors = sum(t != r for t, r in zip(transmitted_bits, received_bits))
    total_bits = len(transmitted_bits)
    ber = bit_errors / total_bits
    return ber, bit_errors, total_bits

# Function to calculate BER for a given SNR
def calculate_ber_for_snr(SNR_dB):
    # Calculate noise level based on SNR
    signal_power = np.mean(np.abs(ofdm_time_signal_cp)**2)
    SNR_linear = 10**(SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power / 2) * (np.random.normal(0, 1, len(ofdm_time_signal_cp)) + 1j * np.random.normal(0, 1, len(ofdm_time_signal_cp)))

    # Transmit through noisy channel
    received_signal = ofdm_time_signal_cp + noise

    # Remove cyclic prefix from each OFDM symbol
    received_signal = received_signal[cp_length:]

    # OFDM demodulation
    received_symbols = fft(received_signal)

    # Separate data symbols from pilots
    received_qam_symbols = received_symbols[data_indices[:len(qam_symbols)]]

    # QAM demodulation
    received_binary_data = qam_demodulate(received_qam_symbols, qam_order)

    # Calculate BER
    ber, bit_errors, total_bits = calculate_ber(binary_data, received_binary_data)
    return ber

# Parameters for a realistic 2.4 GHz WiFi OFDM example
channel_bandwidth = 20e6  # 20 MHz channel
num_subcarriers = 64  # Number of OFDM subcarriers
subcarrier_spacing = channel_bandwidth / num_subcarriers  # Subcarrier spacing: 312.5 kHz
Fs = channel_bandwidth  # Sampling frequency
center_frequency = 2.4e9  # Center frequency for the 2.4 GHz band (e.g., Channel 1)
qam_order = 16  # QAM order (e.g., 4 for QPSK, 16 for 16-QAM, 64 for 64-QAM)
SNR_dB = 14  # Signal-to-Noise Ratio in dB
pilot_pos = 8  # Pilot position in the OFDM frame

# Pilot parameters
pilot_symbol = 2 + 2j  # Example pilot symbol
# pilot_indices = np.arange(num_subcarriers)  # Example pilot indices in the OFDM subcarriers

# Generate QAM constellation
qam_constellation = generate_qam_constellation(qam_order)

# Generate time vector
T = 1 / subcarrier_spacing  # OFDM symbol duration
t = np.arange(0, T, 1/Fs)

# Convert string to binary
data = "1234567812345678"
binary_data = string_to_binary(data)

# QAM modulation
qam_symbols = qam_modulate(binary_data, qam_order)

# Insert pilot symbols into the OFDM frame
ofdm_symbols = np.zeros(num_subcarriers, dtype=complex)
ofdm_indices = np.arange(num_subcarriers)
pilot_carriers = ofdm_indices[::num_subcarriers//pilot_pos]
data_indices = [i for i in range(num_subcarriers) if i not in pilot_carriers]
ofdm_symbols[data_indices[:len(qam_symbols)]] = qam_symbols
ofdm_symbols[pilot_carriers] = pilot_symbol

# Ensure the number of QAM symbols does not exceed the number of OFDM subcarriers
if len(qam_symbols) + len(pilot_carriers) > num_subcarriers:
    raise ValueError("Number of QAM symbols and pilots exceeds the number of OFDM subcarriers")

# OFDM modulation
ofdm_time_signal = ifft(ofdm_symbols)

# Add cyclic prefix (CP) to each OFDM symbol
cp_length = num_subcarriers // 4  # Length of cyclic prefix
ofdm_time_signal_cp = np.concatenate([ofdm_time_signal[-cp_length:], ofdm_time_signal])

# Calculate noise level based on SNR
signal_power = np.mean(np.abs(ofdm_time_signal_cp)**2)
SNR_linear = 10**(SNR_dB / 10)
noise_power = signal_power / SNR_linear
noise = np.sqrt(noise_power / 2) * (np.random.normal(-1, 1, len(ofdm_time_signal_cp)) + 1j * np.random.normal(-1, 1, len(ofdm_time_signal_cp)))

# Transmit through noisy channel
received_signal = ofdm_time_signal_cp + noise

recieved_signal_with_cp = received_signal

# Remove cyclic prefix from each OFDM symbol
received_signal = received_signal[cp_length:]

# OFDM demodulation
received_symbols = fft(received_signal)

# Separate data symbols from pilots
received_qam_symbols = received_symbols[data_indices[:len(qam_symbols)]]

# QAM demodulation
received_binary_data = qam_demodulate(received_qam_symbols, qam_order)

# Convert binary to string
received_data = binary_to_string(received_binary_data)

# Calculate BER
ber, bit_errors, total_bits = calculate_ber(binary_data, received_binary_data)

# Frequency axis for the OFDM spectrum
freqs = np.fft.fftfreq(num_subcarriers, 1/Fs) + center_frequency

# Display results
print("Original data:", data)
print("Binary data:", binary_data)
print("Received binary data:", received_binary_data)
print("Received data:", received_data)
print(f"BER: {ber:.2e}, Bit Errors: {bit_errors}, Total Bits: {total_bits}")

# Print the frequencies and symbols of each subcarrier
print("Frequencies and symbols of each subcarrier:")
for i, (freq, symbol) in enumerate(zip(freqs[:num_subcarriers//2], ofdm_symbols[:num_subcarriers//2])):
    print(f"Subcarrier {i+1}: Frequency {freq:.2e} Hz, Symbol {symbol}")

# Plot the QAM constellation for transmitted symbols
plt.figure(figsize=(12, 6))
plt.scatter(np.real(qam_constellation), np.imag(qam_constellation), color='blue')
for i, point in enumerate(qam_constellation):
    plt.text(np.real(point), np.imag(point), format(i, '0' + str(int(np.log2(qam_order))) + 'b'),
             fontsize=12, ha='center', va='center', color='black')
plt.scatter(np.real(qam_symbols), np.imag(qam_symbols), color='red', marker='x', label='Transmitted Symbols')
plt.title(f'{qam_order}-QAM Constellation - Transmitted Symbols')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# Plot the QAM constellation for received symbols
plt.figure(figsize=(12, 6))
plt.scatter(np.real(qam_constellation), np.imag(qam_constellation), color='blue')
for i, point in enumerate(qam_constellation):
    plt.text(np.real(point), np.imag(point), format(i, '0' + str(int(np.log2(qam_order))) + 'b'),
             fontsize=12, ha='center', va='center', color='black')
plt.scatter(np.real(received_qam_symbols), np.imag(received_qam_symbols), color='green', marker='o', label='Received Symbols')
plt.title(f'{qam_order}-QAM Constellation - Received Symbols')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# Plot the signals and spectrum
plt.figure(figsize=(14, 10))

# Plot the transmitted OFDM signal in time domain
plt.subplot(4, 1, 1)
plt.plot(np.real(ofdm_time_signal_cp), label='Transmitted OFDM Signal (Real)')
plt.plot(np.imag(ofdm_time_signal_cp), label='Transmitted OFDM Signal (Imag)', linestyle='--')
plt.plot(np.abs(ofdm_time_signal_cp), label='Transmitted OFDM Signal (Abs)', linestyle='-')
plt.title('Transmitted OFDM Signal in Time Domain')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.legend()


# Plot the received OFDM signal in time domain
plt.subplot(4, 1, 2)
plt.plot(np.real(recieved_signal_with_cp), label='Received OFDM Signal with CP (Real)')
plt.plot(np.imag(recieved_signal_with_cp), label='Received OFDM Signal with CP (Imag)', linestyle='--')
plt.plot(np.abs(recieved_signal_with_cp), label='Received OFDM Signal with CP (Abs)', linestyle='-')
plt.title('Received OFDM Signal with CP in Time Domain')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.legend()

# Plot the received OFDM signal in time domain
plt.subplot(4, 1, 3)
plt.plot(np.real(received_signal), label='Received OFDM Signal (Real)')
plt.plot(np.imag(received_signal), label='Received OFDM Signal (Imag)', linestyle='--')
plt.plot(np.abs(received_signal), label='Received OFDM Signal (Abs)', linestyle='-')
plt.title('Received OFDM Signal in Time Domain')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.legend()

# Plot the frequency spectrum of the OFDM symbols, highlighting pilots
plt.subplot(4, 1, 4)
plt.stem(freqs[:num_subcarriers//2], np.abs(ofdm_symbols[:num_subcarriers//2]), basefmt=" ", label='Data Subcarriers')
plt.scatter(np.array(pilot_carriers) * subcarrier_spacing + center_frequency, np.abs(ofdm_symbols[pilot_carriers]),color = 'red', marker='x', label='Pilot Subcarriers')
plt.title('Frequency Spectrum of OFDM Symbols')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the sinc functions for each subcarrier frequency in the frequency domain
plt.figure(figsize=(12, 8))
freq_resolution = np.linspace(center_frequency - channel_bandwidth / 2, center_frequency + channel_bandwidth / 2, 1000)
for i, freq in enumerate(freqs[:num_subcarriers//2]):
    if np.abs(ofdm_symbols[i]) > 0:  # Only plot if the subcarrier has a signal
        sinc_func = np.sinc((freq_resolution - freq) / subcarrier_spacing)
        plt.plot(freq_resolution, sinc_func) #, label=f'Subcarrier {i+1}')

plt.title('Sinc Functions for Each Subcarrier Frequency in Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
# plt.xlim(2.3975e9, center_frequency + num_subcarriers * subcarrier_spacing)
plt.tight_layout()
plt.show()

# Calculate BER for a range of SNR values
SNR_dB_range = np.arange(-30, 30, 0.01)
BER_values = [calculate_ber_for_snr(SNR_dB) for SNR_dB in SNR_dB_range]

# Plot BER vs SNR
plt.figure(figsize=(10, 6))
plt.scatter(SNR_dB_range, BER_values, marker='o')
plt.title('BER vs SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.yscale('log')
plt.grid(True)
plt.show()
