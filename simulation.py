import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import zlib, io, base64, sys, json

# --- Globals that will be set by run_simulation ---
# These are needed because the helper functions rely on them.
K = 64
CP = 16
pilotValue = 1+1j
P = 16
allCarriers = np.arange(K)
pilotCarriers = allCarriers[::P]
dataCarriers = np.delete(allCarriers, pilotCarriers)
mu = 2
mapping_table = {}
demapping_table = {}
payloadBits_per_OFDM = 0

class CapturedOutput:
    def __init__(self):
        self.content = ''
    def write(self, text):
        self.content += text
    def flush(self):
        pass

# --- Hamming (7,4) ---
def hamming_encode(data):
    # ... (encoding part is unchanged) ...
    n = len(data)
    num_blocks = (n + 3) // 4
    padded = np.pad(data, (0, num_blocks*4 - n), constant_values=0)
    encoded = np.zeros(num_blocks*7, dtype=np.uint8)
    for i in range(num_blocks):
        d = padded[i*4:(i+1)*4]
        d1,d2,d3,d4 = d
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        enc = [p1,p2,d1,p3,d2,d3,d4]
        encoded[i*7:(i+1)*7] = enc
    return encoded

def hamming_decode(encoded):
    n = len(encoded)
    num_blocks = n // 7
    decoded = np.zeros(num_blocks*4, dtype=np.uint8)
    corrected_bits = 0 # FIXED: Initialize counter
    for i in range(num_blocks):
        r = encoded[i*7:(i+1)*7].copy()
        s1 = r[0] ^ r[2] ^ r[4] ^ r[6]
        s2 = r[1] ^ r[2] ^ r[5] ^ r[6]
        s3 = r[3] ^ r[4] ^ r[5] ^ r[6]
        syndrome = s3*4 + s2*2 + s1
        if syndrome != 0:
                pos = syndrome - 1
                if 0 <= pos < 7:
                    r[pos] ^= 1
                    corrected_bits += 1 # This line is correct
        decoded[i*4:(i+1)*4] = r[[2,4,5,6]]
    return decoded, corrected_bits # MODIFIED

# --- RS(255,223) Implementation (GF(2^8)) ---
# ... (All RS helper functions init_gf, gf_mul, poly_mul, etc. are unchanged) ...
RS_N = 255
RS_K = 223
RS_T = (RS_N - RS_K)//2 # 16
GF_EXP = [0]*512
GF_LOG = [0]*256
def init_gf():
    x = 1
    for i in range(255):
        GF_EXP[i] = x
        GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11d
    for i in range(255,512):
        GF_EXP[i] = GF_EXP[i-255]
init_gf()
def gf_add(a,b): return a ^ b
def gf_sub(a,b): return a ^ b
def gf_mul(a,b):
    if a == 0 or b == 0: return 0
    return GF_EXP[(GF_LOG[a] + GF_LOG[b]) % 255]
def gf_div(a,b):
    if b == 0: raise ZeroDivisionError()
    if a == 0: return 0
    return GF_EXP[(GF_LOG[a] - GF_LOG[b]) % 255]
def gf_pow(a, power):
    if a == 0: return 0
    return GF_EXP[(GF_LOG[a]*power) % 255]
def gf_inverse(a):
    if a == 0: raise ZeroDivisionError()
    return GF_EXP[255 - GF_LOG[a]]
def poly_scale(p, x): return [gf_mul(c, x) for c in p]
def poly_add(p, q):
    if len(p) < len(q): p, q = q, p
    res = p.copy()
    offset = len(p) - len(q)
    for i, val in enumerate(q): res[i+offset] ^= val
    return res
def poly_mul(p, q):
    r = [0]*(len(p)+len(q)-1)
    for i,a in enumerate(p):
        if a==0: continue
        for j,b in enumerate(q):
            if b==0: continue
            r[i+j] ^= gf_mul(a,b)
    return r
def rs_generator_poly():
    g = [1]
    for i in range(1, 2*RS_T+1):
        g = poly_mul(g, [1, GF_EXP[i]])
    return g
GEN = rs_generator_poly()
def rs_encode_block(data_bytes):
    if len(data_bytes) != RS_K: raise ValueError("rs_encode_block expects length RS_K")
    msg = list(data_bytes) + [0]*(RS_N - RS_K)
    for i in range(RS_K):
        coef = msg[i]
        if coef != 0:
            for j in range(len(GEN)-1):
                msg[i+j+1] ^= gf_mul(GEN[j+1], coef)
    parity = msg[RS_K:RS_N]
    return list(data_bytes) + parity
def rs_compute_syndromes(rx):
    synd = []
    for j in range(1, 2*RS_T + 1):
        s = 0
        for i, val in enumerate(rx):
            if val != 0:
                s ^= gf_mul(val, GF_EXP[(j * i) % 255])
        synd.append(s)
    return synd
def berlekamp_massey(synd):
    C = [1] + [0]*(2*RS_T); B = [1] + [0]*(2*RS_T); L = 0; m = 1; b = 1
    for n in range(0, 2*RS_T):
        d = synd[n]
        for i in range(1, L+1): d ^= gf_mul(C[i], synd[n - i])
        if d == 0: m += 1
        else:
            T = C.copy(); coef = gf_div(d, b)
            for i in range(0, 2*RS_T+1 - m):
                if i + m < len(C) and i < len(B): C[i + m] ^= gf_mul(coef, B[i])
            if 2*L <= n:
                L_new = n + 1 - L; B = T; b = d; L = n + 1 - L; m = 1
            else: m += 1
    return C[:L+1]
def chien_search(locator):
    errs = []; L = len(locator)
    for i in range(RS_N):
        x_inv = GF_EXP[(255 - i) % 255]; val = 0
        for j in range(L - 1, -1, -1): val = gf_mul(val, x_inv) ^ locator[j]
        if val == 0: errs.append(i)
    if len(errs) != (L - 1): return []
    return errs
def forney(omega, locator, error_positions):
    error_values = []; L = len(locator)
    for pos in error_positions:
        x_inv = GF_EXP[(255 - pos) % 255]; num = 0
        for i in range(len(omega)): num ^= gf_mul(omega[i], gf_pow(x_inv, i))
        denom = 0
        for i in range(1, L):
            if i % 2 == 1: denom ^= gf_mul(locator[i], gf_pow(x_inv, i - 1))
        if denom == 0: return []
        err_val = gf_div(num, denom)
        error_values.append(err_val)
    return error_values

def rs_correct_block(rx_block):
    if len(rx_block) != RS_N: raise ValueError("rs_correct_block expects length RS_N")
    
    # --- FIX: CONVENTION MISMATCH ---
    # The encoder (rs_encode_block) is MSB-first, but all decoder
    # helper functions (syndromes, BM, chien, forney) are LSB-first.
    # We must operate on a *reversed* (LSB-first) copy of the block.
    rx_rev = list(reversed(rx_block))

    # 1. Compute syndromes on the LSB-first block
    synd = rs_compute_syndromes(rx_rev)
    
    # If no syndromes, block is clean. Return original MSB-first data.
    if not any(synd): 
        return rx_block[:RS_K], 0 # MODIFIED (0 corrections)
    
    # 2. Find error locator polynomial (LSB-first)
    locator = berlekamp_massey(synd)
    if len(locator) - 1 > RS_T: 
        return rx_block[:RS_K], -1 # MODIFIED (failure)
        
    # 3. Find error positions (LSB-first indices)
    error_positions = chien_search(locator)
    if not error_positions or len(error_positions) != len(locator) - 1:
        return rx_block[:RS_K], -1 # MODIFIED (failure)

    # 4. Find error evaluator polynomial (Omega)
    # The original code's method for LSB-first poly_mul was convoluted but correct:
    # omega_poly = list(reversed(poly_mul(list(reversed(locator)), list(reversed(synd)))))
    # This is equivalent to poly_mul_LSB(locator, synd)
    S_poly = synd 
    omega_poly = poly_mul(list(reversed(locator)), list(reversed(S_poly)))
    omega_poly = list(reversed(omega_poly))
    omega = ([0] * (2*RS_T)) # Truncate to 2T terms
    for i in range(min(len(omega_poly), 2*RS_T)): 
        omega[i] = omega_poly[i]

    # 5. Find error magnitudes
    error_magnitudes = forney(omega, locator, error_positions)
    if not error_magnitudes: 
        return rx_block[:RS_K], -1 # MODIFIED (failure)

    # 6. Apply corrections to the LSB-FIRST block
    corrected_rev = rx_rev.copy()
    for i in range(len(error_positions)):
        pos = error_positions[i] # This is an LSB-first index
        val = error_magnitudes[i]
        
        if pos < len(corrected_rev): 
            corrected_rev[pos] ^= val # Apply fix to LSB-first block
        
    # 7. Verify correction on the LSB-FIRST block
    if any(rs_compute_syndromes(corrected_rev)):
        return rx_block[:RS_K], -1 # MODIFIED (failure)
        
    # 8. --- FIX: --- Reverse the corrected block back to MSB-first
    corrected_block = list(reversed(corrected_rev))
    
    # Return the data portion of the original (MSB-first) corrected block
    return corrected_block[:RS_K], len(error_positions) # MODIFIED (success)

# --- Helper: bytes/bits conversions ---
# ... (bits_to_bytes and bytes_to_bits are unchanged) ...
def bits_to_bytes(bits):
    return np.packbits(np.array(bits, dtype=np.uint8))
def bytes_to_bits(arr, desired_bits=None):
    b = np.unpackbits(np.array(arr, dtype=np.uint8))
    if desired_bits is not None:
        return b[:desired_bits]
    return b

# --- OFDM functions ---
# ... (SP, Mapping, OFDM_symbol, IDFT, addCP are unchanged) ...
def SP(bits):
    if len(bits) < len(dataCarriers) * mu:
        bits = np.pad(bits, (0, len(dataCarriers) * mu - len(bits)), constant_values=0)
    return bits.reshape((len(dataCarriers), mu))
def Mapping(bits):
    if mu == 1:
        return np.array([mapping_table[tuple(b)] for b in bits])
    else:
        return np.array([mapping_table[tuple(b)] for b in bits])
def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex)
    symbol[pilotCarriers] = pilotValue
    symbol[dataCarriers] = QAM_payload
    return symbol
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)
def addCP(OFDM_time):
    return np.hstack([OFDM_time[-CP:], OFDM_time])

# --- Channel functions ---
# ... (rician_fading, channel, removeCP, DFT, channelEstimate, equalize, get_payload are unchanged) ...
def rician_fading(N, K_db):
    K_lin = 10**(K_db/10.0)
    los = np.sqrt(K_lin/(K_lin+1.0))
    nlos = np.sqrt(1.0/(2*(K_lin+1.0))) * (np.random.randn() + 1j*np.random.randn())
    phi = np.exp(1j * 2*np.pi*np.random.rand())
    h = (los*phi + nlos) * np.ones(N)
    return h
def channel(signal, SNR_db, K_db, doppler_norm, channelResponse):
    convolved = np.convolve(signal, channelResponse, mode='same')
    h = rician_fading(len(convolved), K_db)
    faded = convolved * h
    t = np.arange(len(faded))
    doppler_shift = np.exp(1j * 2*np.pi * doppler_norm * t)
    faded = faded * doppler_shift
    signal_power = np.mean(np.abs(faded)**2)
    sigma2 = signal_power * 10**(-SNR_db/10.0)
    noise = np.sqrt(sigma2/2.0) * (np.random.randn(len(faded)) + 1j*np.random.randn(len(faded)))
    return faded + noise
def removeCP(signal):
    return signal[CP:(CP+K)]
def DFT(x):
    return np.fft.fft(x)
def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]
    Hpil = pilots / pilotValue
    kind = 'cubic' if len(pilotCarriers) >= 4 else 'linear'
    interp_real = interp1d(pilotCarriers, Hpil.real, kind=kind, fill_value='extrapolate')
    interp_imag = interp1d(pilotCarriers, Hpil.imag, kind=kind, fill_value='extrapolate')
    Hreal = interp_real(np.arange(K))
    Himag = interp_imag(np.arange(K))
    H = Hreal + 1j*Himag
    win = 3
    if win > 1:
        kernel = np.ones(win) / win
        H = np.convolve(H, kernel, mode='same')
    return H
def equalize(OFDM_demod, H):
    return OFDM_demod / H
def get_payload(equalized):
    return equalized[dataCarriers]

# ... (Demapping, MAC, calculate_data_rate are unchanged) ...
def Demapping(QAM):
    constellation_values = np.array(list(demapping_table.keys()))
    QAM_power = np.mean(np.abs(QAM)**2)
    const_power = np.mean(np.abs(constellation_values)**2)
    QAM_scaled = QAM * np.sqrt(const_power / QAM_power) if QAM_power > 0 else QAM
    dists = np.abs(QAM_scaled.reshape((-1,1)) - constellation_values.reshape((1,-1)))
    const_index = dists.argmin(axis=1)
    hardDecision = constellation_values[const_index]
    bits = np.vstack([demapping_table[c] for c in hardDecision])
    return bits, hardDecision
def mac_encapsulate(bits):
    length = len(bits)
    header = np.array([int(b) for b in bin(length)[2:].zfill(16)], dtype=np.uint8)
    data_with_header = np.hstack([header, bits])
    crc = zlib.crc32(data_with_header.tobytes())
    crc_bits = np.array([int(b) for b in bin(crc)[2:].zfill(32)], dtype=np.uint8)
    return np.hstack([data_with_header, crc_bits])
def mac_decapsulate(bits_est):
    if len(bits_est) < 48:
        return None, "Packet too short"
    header = bits_est[:16]
    length = int(''.join(map(str, header)), 2)
    expected_len = 16 + length + 32
    if len(bits_est) < expected_len:
        return None, "Incomplete packet based on header"
    data = bits_est[16:16+length]
    crc_est_bits = bits_est[16+length:16+length+32]
    crc_calc = zlib.crc32(bits_est[:16+length].tobytes())
    crc_calc_bits_str = bin(crc_calc)[2:].zfill(32)
    if ''.join(map(str, crc_est_bits)) != crc_calc_bits_str:
        return data, "CRC Mismatch Error!"
    return data, None
def calculate_data_rate(K, CP, dataCarriers, mu, fec_type, RS_K, RS_N, symbol_rate_msym):
    modulation_rate = len(dataCarriers) * mu
    ofdm_efficiency = K / (K + CP)
    if fec_type == "hamming":
        coding_rate = 4.0 / 7.0
    elif fec_type == "reed_solomon":
        coding_rate = RS_K / RS_N
    else:
        coding_rate = 1.0
    symbol_rate_s = symbol_rate_msym * 1e6
    data_rate = symbol_rate_s * modulation_rate * ofdm_efficiency * coding_rate
    return data_rate

# --- Main simulation ---
def run_simulation(input_text, SNR_db, K_db_param, doppler_norm_param, K_subcarriers_param, fec_type, mod_scheme, symbol_rate_msym):
    global K, CP, pilotCarriers, dataCarriers, payloadBits_per_OFDM, mu, mapping_table, demapping_table, P, pilotValue

    # This state needs to be set for the helper functions to work
    channelResponse = np.array([1])
    
    # --- Redefine modulation and OFDM params based on inputs ---
    if mod_scheme == "BPSK":
        mu = 1
        mapping_table = {
            (0,): -1.0, 
            (1,): 1.0
        }
    elif mod_scheme == "QPSK":
        mu = 2
        factor = 1.0/np.sqrt(2) 
        mapping_table = {
            (0,0): factor * (-1-1j), (0,1): factor * (-1+1j),
            (1,0): factor * (1-1j), (1,1): factor * (1+1j)
        }
    elif mod_scheme == "16QAM":
        mu = 4
        factor = 1.0/np.sqrt(10)
        mapping_table = {
            (0,0,0,0): factor * (-3-3j), (0,0,0,1): factor * (-3-1j), (0,0,1,1): factor * (-3+1j), (0,0,1,0): factor * (-3+3j),
            (0,1,0,0): factor * (-1-3j), (0,1,0,1): factor * (-1-1j), (0,1,1,1): factor * (-1+1j), (0,1,1,0): factor * (-1+3j),
            (1,1,0,0): factor * (1-3j), (1,1,0,1): factor * (1-1j), (1,1,1,1): factor * (1+1j), (1,1,1,0): factor * (1+3j),
            (1,0,0,0): factor * (3-3j), (1,0,0,1): factor * (3-1j), (1,0,1,1): factor * (3+1j), (1,0,1,0): factor * (3+3j)
        }
    elif mod_scheme == "64QAM":
        mu = 6
        factor = 1.0 / np.sqrt(42)
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        gray_map = [0b000, 0b001, 0b011, 0b010, 0b110, 0b111, 0b101, 0b100]
        mapping_table = {}
        for pos_i in range(8):
            gray_int_i = gray_map[pos_i]
            bits_i = ((gray_int_i >> 2) & 1, (gray_int_i >> 1) & 1, gray_int_i & 1)
            for pos_j in range(8):
                gray_int_j = gray_map[pos_j]
                bits_j = ((gray_int_j >> 2) & 1, (gray_int_j >> 1) & 1, gray_int_j & 1)
                bits = bits_i + bits_j
                I = levels[pos_i]
                Q = levels[pos_j]
                mapping_table[bits] = factor * (I + 1j * Q)
    else: # Default to QPSK
        mu = 2
        factor = 1.0/np.sqrt(2)
        mapping_table = {
            (0,0): factor * (-1-1j), (0,1): factor * (-1+1j),
            (1,0): factor * (1-1j), (1,1): factor * (1+1j)
        }
    demapping_table = {v:k for k,v in mapping_table.items()}

    try:
        K_val = int(K_subcarriers_param)
        K = K_val if K_val>0 and (K_val & (K_val-1))==0 else 64
    except:
        K = 64
        
    CP = K//4
    P = max(1, K // 4)
    allCarriers = np.arange(K)
    pilotCarriers = allCarriers[::P]
    if allCarriers[-1] not in pilotCarriers:
        pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
    dataCarriers = np.delete(allCarriers, pilotCarriers)
    if len(pilotCarriers) < 4:
        step = max(1, K // 4)
        pilotCarriers = allCarriers[::step]
        if allCarriers[-1] not in pilotCarriers:
            pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
        dataCarriers = np.delete(allCarriers, pilotCarriers)

    payloadBits_per_OFDM = len(dataCarriers)*mu
    
    try:
        K_db = float(K_db_param)
    except:
        K_db = 10.0
    try:
        doppler_norm = float(doppler_norm_param)
    except:
        doppler_norm = 0.001

    # ... (Data rate calculation is unchanged) ...
    data_rate_bps = calculate_data_rate(K, CP, dataCarriers, mu, fec_type, RS_K, RS_N, symbol_rate_msym)
    if data_rate_bps >= 1e9:
        rate_str = f"{data_rate_bps/1e9:.2f} Gbps"
    elif data_rate_bps >= 1e6:
        rate_str = f"{data_rate_bps/1e6:.2f} Mbps"
    else:
        rate_str = f"{data_rate_bps:.0f} bps"

    captured = CapturedOutput()
    sys.stdout = captured

    # ... (Printing config is unchanged) ...
    print('--- Configuration ---')
    print(f'Symbol Rate (F_s): {symbol_rate_msym} MSym/s')
    print(f'Modulation: {mod_scheme} ({mu} bits/sym)')
    print(f'Subcarriers (K): {K} | CP: {CP} (Efficiency: {K/(K+CP):.3f})')
    print(f'Data Carriers: {len(dataCarriers)} | Pilots: {len(pilotCarriers)}')
    print(f'Rician K-factor (dB): {K_db} | Doppler: {doppler_norm}')
    print(f'FEC: {fec_type} (Rate: {4/7:.3f} for Hamming, {RS_K/RS_N:.3f} for RS)')
    print('---------------------')

    # ... (Text to bits and MAC encapsulation is unchanged) ...
    bits = []
    for ch in input_text:
        b = bin(ord(ch))[2:].zfill(8)
        bits.extend([int(x) for x in b])
    bits = np.array(bits, dtype=np.uint8)
    bits_encap = mac_encapsulate(bits)

    # FEC encoding
    # FIXED: Removed total_corrected and fec_failed from here
    if fec_type == "reed_solomon":
        data_bytes = bits_to_bytes(bits_encap)
        blocks = []
        for i in range(0, len(data_bytes), RS_K):
            chunk = data_bytes[i:i+RS_K].tolist()
            if len(chunk) < RS_K:
                chunk = chunk + [0]*(RS_K - len(chunk))
            enc = rs_encode_block(chunk)
            blocks.extend(enc)
        encoded_bytes = np.array(blocks, dtype=np.uint8)
        bits_coded = bytes_to_bits(encoded_bytes)
    else:
        bits_coded = hamming_encode(bits_encap)

    # transmit via OFDM symbol by symbol
    num_symbols = int(np.ceil(len(bits_coded) / payloadBits_per_OFDM))
    bits_padded = np.pad(bits_coded, (0, num_symbols*payloadBits_per_OFDM - len(bits_coded)))
    bits_est_all = np.array([], dtype=np.uint8)
    last_QAM_est = None
    last_hardDecision = None
    last_Hest = None # NEW: Initialize last_Hest

    for i in range(num_symbols):
        # ... (OFDM modulation is unchanged) ...
        bits_sym = bits_padded[i*payloadBits_per_OFDM:(i+1)*payloadBits_per_OFDM]
        bits_SP = SP(bits_sym)
        QAM = Mapping(bits_SP)
        OFDM_data = OFDM_symbol(QAM)
        OFDM_time = IDFT(OFDM_data)
        OFDM_withCP = addCP(OFDM_time)
        
        # ... (Channel and reception is unchanged) ...
        OFDM_RX = channel(OFDM_withCP, SNR_db, K_db, doppler_norm, channelResponse)
        OFDM_RX_comp = OFDM_RX
        OFDM_RX_noCP = removeCP(OFDM_RX_comp)
        OFDM_demod = DFT(OFDM_RX_noCP)
        Hest = channelEstimate(OFDM_demod)
        equalized = equalize(OFDM_demod, Hest)
        QAM_est = get_payload(equalized)
        bits_est_sym, hardDecision = Demapping(QAM_est)
        bits_est_all = np.hstack([bits_est_all, bits_est_sym.reshape(-1)])
        
        # ... (Saving last symbol's data is unchanged) ...
        last_QAM_est = QAM_est
        last_hardDecision = hardDecision
        if i == num_symbols - 1:
            last_Hest = Hest


    bits_coded_est = bits_est_all[:len(bits_coded)]
    Hest = last_Hest # This line was correct

    # FEC decode
    total_corrected = 0 # NEW: Initialize counters here
    fec_failed = False    # NEW: Initialize flag here

    if fec_type == "reed_solomon":
        est_bytes = bits_to_bytes(bits_coded_est)
        dec_bytes = []
        for i in range(0, len(est_bytes), RS_N):
            block = est_bytes[i:i+RS_N].tolist()
            if len(block) < RS_N:
                block = block + [0]*(RS_N - len(block))
            data_out, corrections = rs_correct_block(block) # MODIFIED
            if corrections == -1: # MODIFIED
                fec_failed = True
            else:
                total_corrected += corrections # MODIFIED
            dec_bytes.extend(data_out)
            # FIXED: Removed the 'decode_ok' logic, it's redundant with fec_failed
        dec_bytes = np.array(dec_bytes[: (len(bits_encap) // 8)], dtype=np.uint8)
        bits_encap_est = bytes_to_bits(dec_bytes, desired_bits=len(bits_encap))
        bits_encap_est = bits_encap_est[:len(bits_encap)]
    else:
        bits_encap_est, total_corrected = hamming_decode(bits_coded_est) # MODIFIED
        bits_encap_est = bits_encap_est[:len(bits_encap)]

    bits_decap, err_msg = mac_decapsulate(bits_encap_est)
    if err_msg:
        fec_failed = True # If CRC fails, FEC effectively failed

    # ... (Printing results is unchanged) ...
    print('--- Simulation Results ---')
    print(f"Original Message: '{input_text}'")
    if err_msg:
        print(f"Decoded Message: FAILED! ({err_msg})")
        out_text = ''
        if (bits_decap is not None):
            for i in range(len(bits_decap)//8):
                byte = bits_decap[i*8:(i+1)*8]
                out_text += chr(int(''.join(map(str, byte)), 2))
            print(f"Bad message: '{out_text}'")
        else:
            print("Message to messed up to decode")
    else:
        out_text = ''
        for i in range(len(bits_decap)//8):
            byte = bits_decap[i*8:(i+1)*8]
            out_text += chr(int(''.join(map(str, byte)), 2))
        print(f"Decoded Message: '{out_text}'")

    ber = np.sum(bits_encap != bits_encap_est) / len(bits_encap)
    print(f"Bit Error Rate (BER): {ber:.6f}")

    # plot
    # ... (Constellation plot generation is unchanged) ...
    fig, ax = plt.subplots(figsize=(6,6))
    if last_QAM_est is not None:
        if mod_scheme == "BPSK":
            ax.plot(last_QAM_est.real, np.zeros_like(last_QAM_est.real), '.', alpha=0.5, label='Received')
            ax.plot(last_hardDecision.real, np.zeros_like(last_hardDecision.real), 'ro', markersize=6, label='Decided')
        else:
            ax.plot(last_QAM_est.real, last_QAM_est.imag, '.', alpha=0.5, label='Received')
    ax.plot(last_hardDecision.real, last_hardDecision.imag, 'ro', markersize=6, label='Decided')
    ax.grid(True)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title(f'{mod_scheme} Constellation (SNR={SNR_db} dB, K={K_db} dB)')
    ax.legend()
    max_const_val = np.max(np.abs(list(mapping_table.values())))
    max_plot_val = max_const_val * 1.5 
    if mod_scheme == "BPSK":
        ax.set_xlim([-max_plot_val, max_plot_val])
        ax.set_ylim([-max_plot_val, max_plot_val])
        ax.set_aspect('equal')
    else:
        ax.set_xlim([-max_plot_val, max_plot_val])
        ax.set_ylim([-max_plot_val, max_plot_val])
        ax.set_aspect('equal')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

    # --- VIZ 3: CHANNEL PLOT (NEW) ---
    fig_ch, ax_ch = plt.subplots(figsize=(6, 4))
    if 'Hest' in locals() and Hest is not None:
        Hest_gain_dB = 20 * np.log10(np.abs(Hest))
        ax_ch.plot(np.arange(K), Hest_gain_dB, 'b-')
        ax_ch.set_title(f'Est. Channel Response (K={K_db} dB)')
        ax_ch.set_xlabel('Subcarrier Index')
        ax_ch.set_ylabel('Gain (dB)')
        ax_ch.grid(True)
        ax_ch.set_ylim([np.min(Hest_gain_dB) - 5, np.max(Hest_gain_dB) + 5])
    else:
        ax_ch.text(0.5, 0.5, 'Channel plot not available.', horizontalalignment='center', verticalalignment='center')
    buf_ch = io.BytesIO()
    fig_ch.savefig(buf_ch, format='png', bbox_inches='tight')
    buf_ch.seek(0)
    plot_ch_b64 = base64.b64encode(buf_ch.read()).decode('utf-8')

    # --- VIZ 4: FEC STATS (NEW) ---
    fec_stats = {
        "corrected": total_corrected,
        "failed": fec_failed
    }
    fec_stats_json = json.dumps(fec_stats)

    sys.stdout = sys.__stdout__
    
    # --- NEW RETURN STATEMENT ---
    return ( captured.content, plot_b64, rate_str, plot_ch_b64, fec_stats_json, 
             pilotCarriers.tolist(), dataCarriers.tolist(), mod_scheme )