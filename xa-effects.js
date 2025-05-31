/**
 * Web Audio API Effects Library
 * Port of librosa.effects Python module
 * 
 * Provides harmonic-percussive source separation, time stretching,
 * pitch shifting, and other audio effects using Web Audio API
 */

class LibrosaEffects {
    constructor() {
        this.audioContext = null;
    }

    /**
     * Initialize the audio context
     */
    init() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        return this.audioContext;
    }

    /**
     * Helper function to compute STFT (Short-Time Fourier Transform)
     */
    stft(y, nFft = 2048, hopLength = null, winLength = null, window = 'hann', center = true) {
        if (!hopLength) hopLength = Math.floor(nFft / 4);
        if (!winLength) winLength = nFft;

        // Create window function
        const win = this.getWindow(window, winLength);
        
        // Pad signal if center is true
        let paddedY = y;
        if (center) {
            const padLength = Math.floor(nFft / 2);
            paddedY = new Float32Array(y.length + 2 * padLength);
            paddedY.set(y, padLength);
        }

        // Calculate number of frames
        const nFrames = Math.floor((paddedY.length - nFft) / hopLength) + 1;
        
        // Initialize STFT matrix (complex values)
        const stftMatrix = {
            real: new Array(Math.floor(nFft / 2) + 1).fill(0).map(() => new Float32Array(nFrames)),
            imag: new Array(Math.floor(nFft / 2) + 1).fill(0).map(() => new Float32Array(nFrames))
        };

        // Compute STFT
        for (let t = 0; t < nFrames; t++) {
            const start = t * hopLength;
            const frame = paddedY.slice(start, start + nFft);
            
            // Apply window
            const windowedFrame = new Float32Array(nFft);
            for (let i = 0; i < winLength; i++) {
                windowedFrame[i] = frame[i] * win[i];
            }

            // FFT
            const fftResult = this.fft(windowedFrame);
            
            // Store positive frequencies only
            for (let f = 0; f <= nFft / 2; f++) {
                stftMatrix.real[f][t] = fftResult.real[f];
                stftMatrix.imag[f][t] = fftResult.imag[f];
            }
        }

        return stftMatrix;
    }

    /**
     * Helper function to compute ISTFT (Inverse Short-Time Fourier Transform)
     */
    istft(stftMatrix, hopLength = null, winLength = null, window = 'hann', center = true, length = null) {
        const nFft = (stftMatrix.real.length - 1) * 2;
        if (!hopLength) hopLength = Math.floor(nFft / 4);
        if (!winLength) winLength = nFft;

        const win = this.getWindow(window, winLength);
        const nFrames = stftMatrix.real[0].length;
        
        // Calculate output length
        let expectedLength = nFft + hopLength * (nFrames - 1);
        if (center) {
            expectedLength -= nFft;
        }
        
        const y = new Float32Array(expectedLength);
        const windowSum = new Float32Array(expectedLength);

        // Overlap-add synthesis
        for (let t = 0; t < nFrames; t++) {
            // Reconstruct full spectrum
            const fullSpectrum = {
                real: new Float32Array(nFft),
                imag: new Float32Array(nFft)
            };

            // Positive frequencies
            for (let f = 0; f <= nFft / 2; f++) {
                fullSpectrum.real[f] = stftMatrix.real[f][t];
                fullSpectrum.imag[f] = stftMatrix.imag[f][t];
            }

            // Negative frequencies (complex conjugate)
            for (let f = 1; f < nFft / 2; f++) {
                fullSpectrum.real[nFft - f] = stftMatrix.real[f][t];
                fullSpectrum.imag[nFft - f] = -stftMatrix.imag[f][t];
            }

            // IFFT
            const frame = this.ifft(fullSpectrum);
            
            // Apply window and overlap-add
            const start = t * hopLength;
            for (let i = 0; i < winLength; i++) {
                if (start + i < y.length) {
                    y[start + i] += frame.real[i] * win[i];
                    windowSum[start + i] += win[i] * win[i];
                }
            }
        }

        // Normalize by window sum
        for (let i = 0; i < y.length; i++) {
            if (windowSum[i] > 0) {
                y[i] /= windowSum[i];
            }
        }

        // Trim if center was true
        if (center) {
            const padLength = Math.floor(nFft / 2);
            return y.slice(padLength, padLength + (length || y.length - 2 * padLength));
        }

        return length ? y.slice(0, length) : y;
    }

    /**
     * FFT implementation using Web Audio API's AnalyserNode as inspiration
     * For production, consider using a library like DSP.js or FFTJS
     */
    fft(signal) {
        const N = signal.length;
        const real = new Float32Array(N);
        const imag = new Float32Array(N);

        // Simple DFT implementation (not optimized)
        // In production, use FFT library for better performance
        for (let k = 0; k < N; k++) {
            let sumReal = 0;
            let sumImag = 0;
            for (let n = 0; n < N; n++) {
                const angle = -2 * Math.PI * k * n / N;
                sumReal += signal[n] * Math.cos(angle);
                sumImag += signal[n] * Math.sin(angle);
            }
            real[k] = sumReal;
            imag[k] = sumImag;
        }

        return { real, imag };
    }

    /**
     * IFFT implementation
     */
    ifft(spectrum) {
        const N = spectrum.real.length;
        const real = new Float32Array(N);
        const imag = new Float32Array(N);

        // Simple IDFT implementation (not optimized)
        for (let n = 0; n < N; n++) {
            let sumReal = 0;
            let sumImag = 0;
            for (let k = 0; k < N; k++) {
                const angle = 2 * Math.PI * k * n / N;
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                sumReal += spectrum.real[k] * cos - spectrum.imag[k] * sin;
                sumImag += spectrum.real[k] * sin + spectrum.imag[k] * cos;
            }
            real[n] = sumReal / N;
            imag[n] = sumImag / N;
        }

        return { real, imag };
    }

    /**
     * Generate window function
     */
    getWindow(windowType, length) {
        const window = new Float32Array(length);
        
        switch (windowType) {
            case 'hann':
                for (let i = 0; i < length; i++) {
                    window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (length - 1));
                }
                break;
            case 'hamming':
                for (let i = 0; i < length; i++) {
                    window[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (length - 1));
                }
                break;
            default:
                window.fill(1.0); // Rectangular window
        }
        
        return window;
    }

    /**
     * Median filter for HPSS
     */
    medianFilter(matrix, kernelSize) {
        const [height, width] = [matrix.length, matrix[0].length];
        const filtered = new Array(height).fill(0).map(() => new Float32Array(width));
        const halfKernel = Math.floor(kernelSize / 2);

        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                const values = [];
                
                for (let di = -halfKernel; di <= halfKernel; di++) {
                    const ni = i + di;
                    if (ni >= 0 && ni < height) {
                        values.push(matrix[ni][j]);
                    }
                }
                
                values.sort((a, b) => a - b);
                filtered[i][j] = values[Math.floor(values.length / 2)];
            }
        }

        return filtered;
    }

    /**
     * Harmonic-Percussive Source Separation (HPSS)
     */
    hpss(y, {
        kernelSize = 31,
        power = 2.0,
        mask = false,
        margin = 1.0,
        nFft = 2048,
        hopLength = null,
        winLength = null,
        window = 'hann',
        center = true
    } = {}) {
        // Compute STFT
        const stft = this.stft(y, nFft, hopLength, winLength, window, center);
        
        // Compute magnitude spectrogram
        const magnitude = stft.real.map((row, i) => 
            row.map((val, j) => Math.sqrt(val * val + stft.imag[i][j] * stft.imag[i][j]))
        );

        // Apply power scaling
        const powerSpec = magnitude.map(row => 
            row.map(val => Math.pow(val, power))
        );

        // Median filtering
        const harmonicKernel = Array.isArray(kernelSize) ? kernelSize[0] : kernelSize;
        const percussiveKernel = Array.isArray(kernelSize) ? kernelSize[1] : kernelSize;

        // Harmonic: median filter across frequency axis
        const harmonicEnhanced = this.medianFilter(powerSpec, harmonicKernel);
        
        // Percussive: median filter across time axis (transpose, filter, transpose back)
        const transposed = this.transpose(powerSpec);
        const percussiveEnhancedT = this.medianFilter(transposed, percussiveKernel);
        const percussiveEnhanced = this.transpose(percussiveEnhancedT);

        // Compute masks
        let harmonicMask, percussiveMask;

        if (mask) {
            const marginH = Array.isArray(margin) ? margin[0] : margin;
            const marginP = Array.isArray(margin) ? margin[1] : margin;

            harmonicMask = harmonicEnhanced.map((row, i) =>
                row.map((val, j) => val * marginH > percussiveEnhanced[i][j] ? 1.0 : 0.0)
            );

            percussiveMask = percussiveEnhanced.map((row, i) =>
                row.map((val, j) => val * marginP > harmonicEnhanced[i][j] ? 1.0 : 0.0)
            );
        } else {
            // Soft masks using Wiener filtering
            const total = harmonicEnhanced.map((row, i) =>
                row.map((val, j) => val + percussiveEnhanced[i][j])
            );

            harmonicMask = harmonicEnhanced.map((row, i) =>
                row.map((val, j) => total[i][j] > 0 ? val / total[i][j] : 0.0)
            );

            percussiveMask = percussiveEnhanced.map((row, i) =>
                row.map((val, j) => total[i][j] > 0 ? percussiveEnhanced[i][j] / total[i][j] : 0.0)
            );
        }

        // Apply masks to STFT
        const stftHarmonic = {
            real: stft.real.map((row, i) => row.map((val, j) => val * harmonicMask[i][j])),
            imag: stft.imag.map((row, i) => row.map((val, j) => val * harmonicMask[i][j]))
        };

        const stftPercussive = {
            real: stft.real.map((row, i) => row.map((val, j) => val * percussiveMask[i][j])),
            imag: stft.imag.map((row, i) => row.map((val, j) => val * percussiveMask[i][j]))
        };

        // Inverse STFT
        const yHarmonic = this.istft(stftHarmonic, hopLength, winLength, window, center, y.length);
        const yPercussive = this.istft(stftPercussive, hopLength, winLength, window, center, y.length);

        return { harmonic: yHarmonic, percussive: yPercussive };
    }

    /**
     * Extract harmonic component
     */
    harmonic(y, options = {}) {
        const { harmonic } = this.hpss(y, options);
        return harmonic;
    }

    /**
     * Extract percussive component
     */
    percussive(y, options = {}) {
        const { percussive } = this.hpss(y, options);
        return percussive;
    }

    /**
     * Time stretch using phase vocoder
     */
    timeStretch(y, rate, options = {}) {
        if (rate <= 0) {
            throw new Error('Rate must be a positive number');
        }

        const { nFft = 2048, hopLength = null, winLength = null, window = 'hann', center = true } = options;
        
        // Compute STFT
        const stft = this.stft(y, nFft, hopLength, winLength, window, center);
        
        // Phase vocoder
        const stretchedStft = this.phaseVocoder(stft, rate, hopLength || Math.floor(nFft / 4));
        
        // Predict output length
        const outputLength = Math.round(y.length / rate);
        
        // Inverse STFT
        return this.istft(stretchedStft, hopLength, winLength, window, center, outputLength);
    }

    /**
     * Phase vocoder for time stretching
     */
    phaseVocoder(stft, rate, hopLength) {
        const nBins = stft.real.length;
        const nFrames = stft.real[0].length;
        const nFramesStretched = Math.round(nFrames / rate);
        
        const stretchedStft = {
            real: new Array(nBins).fill(0).map(() => new Float32Array(nFramesStretched)),
            imag: new Array(nBins).fill(0).map(() => new Float32Array(nFramesStretched))
        };

        // Phase accumulator
        const phaseAdvance = new Float32Array(nBins);
        const phaseAccumulator = new Float32Array(nBins);

        // Initialize with first frame
        for (let bin = 0; bin < nBins; bin++) {
            stretchedStft.real[bin][0] = stft.real[bin][0];
            stretchedStft.imag[bin][0] = stft.imag[bin][0];
            phaseAdvance[bin] = 2 * Math.PI * bin * hopLength / (2 * (nBins - 1));
        }

        // Process remaining frames
        for (let t = 1; t < nFramesStretched; t++) {
            const sourceFrame = t * rate;
            const frame0 = Math.floor(sourceFrame);
            const frame1 = Math.min(frame0 + 1, nFrames - 1);
            const alpha = sourceFrame - frame0;

            for (let bin = 0; bin < nBins; bin++) {
                // Interpolate magnitude
                const mag0 = Math.sqrt(stft.real[bin][frame0] ** 2 + stft.imag[bin][frame0] ** 2);
                const mag1 = Math.sqrt(stft.real[bin][frame1] ** 2 + stft.imag[bin][frame1] ** 2);
                const mag = mag0 * (1 - alpha) + mag1 * alpha;

                // Accumulate phase
                phaseAccumulator[bin] += phaseAdvance[bin];

                // Apply magnitude and phase
                stretchedStft.real[bin][t] = mag * Math.cos(phaseAccumulator[bin]);
                stretchedStft.imag[bin][t] = mag * Math.sin(phaseAccumulator[bin]);
            }
        }

        return stretchedStft;
    }

    /**
     * Pitch shift
     */
    async pitchShift(y, sampleRate, nSteps, binsPerOctave = 12, options = {}) {
        if (binsPerOctave <= 0 || !Number.isInteger(binsPerOctave)) {
            throw new Error('binsPerOctave must be a positive integer');
        }

        const rate = Math.pow(2, -nSteps / binsPerOctave);
        
        // Time stretch
        const stretched = this.timeStretch(y, rate, options);
        
        // Resample using Web Audio API
        const resampled = await this.resample(stretched, sampleRate / rate, sampleRate);
        
        // Crop to original length
        return resampled.slice(0, y.length);
    }

    /**
     * Resample audio using Web Audio API OfflineAudioContext
     */
    async resample(y, fromSampleRate, toSampleRate) {
        const scale = toSampleRate / fromSampleRate;
        const outputLength = Math.round(y.length * scale);
        
        // Create offline context
        const offlineContext = new OfflineAudioContext(1, outputLength, toSampleRate);
        
        // Create buffer
        const buffer = offlineContext.createBuffer(1, y.length, fromSampleRate);
        buffer.copyToChannel(y, 0);
        
        // Create source
        const source = offlineContext.createBufferSource();
        source.buffer = buffer;
        source.connect(offlineContext.destination);
        source.start();
        
        // Render
        const renderedBuffer = await offlineContext.startRendering();
        const outputArray = new Float32Array(outputLength);
        renderedBuffer.copyFromChannel(outputArray, 0);
        
        return outputArray;
    }

    /**
     * Remix audio by reordering time intervals
     */
    remix(y, intervals, alignZeros = true) {
        const output = [];
        let zeros = null;

        if (alignZeros) {
            zeros = this.findZeroCrossings(y);
            zeros.push(y.length);
        }

        for (const [start, end] of intervals) {
            let segmentStart = start;
            let segmentEnd = end;

            if (alignZeros && zeros) {
                segmentStart = zeros[this.findClosestIndex(zeros, start)];
                segmentEnd = zeros[this.findClosestIndex(zeros, end)];
            }

            output.push(y.slice(segmentStart, segmentEnd));
        }

        // Concatenate all segments
        const totalLength = output.reduce((sum, segment) => sum + segment.length, 0);
        const result = new Float32Array(totalLength);
        let offset = 0;

        for (const segment of output) {
            result.set(segment, offset);
            offset += segment.length;
        }

        return result;
    }

    /**
     * Find zero crossings in signal
     */
    findZeroCrossings(y) {
        const crossings = [];
        
        for (let i = 1; i < y.length; i++) {
            if ((y[i] >= 0 && y[i - 1] < 0) || (y[i] < 0 && y[i - 1] >= 0)) {
                crossings.push(i);
            }
        }
        
        return crossings;
    }

    /**
     * Find closest index in sorted array
     */
    findClosestIndex(arr, value) {
        let left = 0;
        let right = arr.length - 1;
        
        while (left < right) {
            const mid = Math.floor((left + right) / 2);
            if (arr[mid] < value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        if (left > 0 && Math.abs(arr[left - 1] - value) < Math.abs(arr[left] - value)) {
            return left - 1;
        }
        
        return left;
    }

    /**
     * Compute RMS energy
     */
    rms(y, frameLength = 2048, hopLength = 512) {
        const nFrames = Math.floor((y.length - frameLength) / hopLength) + 1;
        const rmsValues = new Float32Array(nFrames);

        for (let i = 0; i < nFrames; i++) {
            const start = i * hopLength;
            const end = start + frameLength;
            let sum = 0;

            for (let j = start; j < end && j < y.length; j++) {
                sum += y[j] * y[j];
            }

            rmsValues[i] = Math.sqrt(sum / frameLength);
        }

        return rmsValues;
    }

    /**
     * Convert amplitude to decibels
     */
    amplitudeToDb(amplitude, ref = Math.max, topDb = 80) {
        const refValue = typeof ref === 'function' ? ref(...amplitude) : ref;
        const db = amplitude.map(val => 20 * Math.log10(Math.max(1e-10, val / refValue)));
        
        if (topDb !== null) {
            const maxDb = Math.max(...db);
            return db.map(val => Math.max(val, maxDb - topDb));
        }
        
        return db;
    }

    /**
     * Trim silence from audio
     */
    trim(y, {
        topDb = 60,
        ref = Math.max,
        frameLength = 2048,
        hopLength = 512
    } = {}) {
        // Compute RMS
        const rmsValues = this.rms(y, frameLength, hopLength);
        
        // Convert to dB
        const db = this.amplitudeToDb(rmsValues, ref, null);
        
        // Find non-silent frames
        const threshold = -topDb;
        let firstFrame = 0;
        let lastFrame = db.length - 1;

        for (let i = 0; i < db.length; i++) {
            if (db[i] > threshold) {
                firstFrame = i;
                break;
            }
        }

        for (let i = db.length - 1; i >= 0; i--) {
            if (db[i] > threshold) {
                lastFrame = i;
                break;
            }
        }

        // Convert frames to samples
        const start = firstFrame * hopLength;
        const end = Math.min(y.length, (lastFrame + 1) * hopLength);

        return {
            trimmed: y.slice(start, end),
            interval: [start, end]
        };
    }

    /**
     * Split audio into non-silent intervals
     */
    split(y, {
        topDb = 60,
        ref = Math.max,
        frameLength = 2048,
        hopLength = 512
    } = {}) {
        // Compute RMS
        const rmsValues = this.rms(y, frameLength, hopLength);
        
        // Convert to dB
        const db = this.amplitudeToDb(rmsValues, ref, null);
        
        // Find non-silent frames
        const threshold = -topDb;
        const nonSilent = db.map(val => val > threshold);
        
        // Find intervals
        const intervals = [];
        let inInterval = false;
        let start = 0;

        for (let i = 0; i < nonSilent.length; i++) {
            if (nonSilent[i] && !inInterval) {
                start = i * hopLength;
                inInterval = true;
            } else if (!nonSilent[i] && inInterval) {
                const end = Math.min(y.length, i * hopLength);
                intervals.push([start, end]);
                inInterval = false;
            }
        }

        // Handle case where audio ends while still in interval
        if (inInterval) {
            intervals.push([start, y.length]);
        }

        return intervals;
    }

    /**
     * Pre-emphasis filter
     */
    preemphasis(y, coef = 0.97, zi = null, returnZf = false) {
        const output = new Float32Array(y.length);
        
        // Initialize filter state
        let z = zi !== null ? zi : 2 * y[0] - y[1];
        
        // Apply filter
        output[0] = y[0] - coef * z;
        for (let i = 1; i < y.length; i++) {
            output[i] = y[i] - coef * y[i - 1];
        }
        
        const zf = y[y.length - 1];
        
        return returnZf ? { output, zf } : output;
    }

    /**
     * De-emphasis filter (inverse of pre-emphasis)
     */
    deemphasis(y, coef = 0.97, zi = null, returnZf = false) {
        const output = new Float32Array(y.length);
        
        // Initialize filter state
        let z = zi !== null ? zi : 0;
        
        // Apply filter
        for (let i = 0; i < y.length; i++) {
            output[i] = y[i] + coef * z;
            z = output[i];
        }
        
        return returnZf ? { output, zf: z } : output;
    }

    /**
     * Helper function to transpose a 2D array
     */
    transpose(matrix) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const transposed = new Array(cols).fill(0).map(() => new Float32Array(rows));
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        
        return transposed;
    }

    /**
     * Load audio from URL or File
     */
    async loadAudio(urlOrFile) {
        if (!this.audioContext) {
            this.init();
        }

        let arrayBuffer;
        
        if (typeof urlOrFile === 'string') {
            // Load from URL
            const response = await fetch(urlOrFile);
            arrayBuffer = await response.arrayBuffer();
        } else if (urlOrFile instanceof File) {
            // Load from File object
            arrayBuffer = await urlOrFile.arrayBuffer();
        } else {
            throw new Error('Input must be a URL string or File object');
        }

        // Decode audio
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        
        // Extract first channel as Float32Array
        const audioData = new Float32Array(audioBuffer.length);
        audioBuffer.copyFromChannel(audioData, 0);
        
        return {
            data: audioData,
            sampleRate: audioBuffer.sampleRate,
            duration: audioBuffer.duration,
            numberOfChannels: audioBuffer.numberOfChannels
        };
    }

    /**
     * Play audio data
     */
    playAudio(audioData, sampleRate = 44100) {
        if (!this.audioContext) {
            this.init();
        }

        const buffer = this.audioContext.createBuffer(1, audioData.length, sampleRate);
        buffer.copyToChannel(audioData, 0);

        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);
        source.start();

        return source;
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LibrosaEffects;
} else {
    window.LibrosaEffects = LibrosaEffects;
}

// Example usage:
/*
const effects = new LibrosaEffects();
effects.init();

// Load audio
const audio = await effects.loadAudio('path/to/audio.mp3');

// Apply HPSS
const { harmonic, percussive } = effects.hpss(audio.data);

// Time stretch
const stretched = effects.timeStretch(audio.data, 0.5); // Half speed

// Pitch shift
const pitched = await effects.pitchShift(audio.data, audio.sampleRate, 4); // Up 4 semitones

// Trim silence
const { trimmed } = effects.trim(audio.data);

// Play result
effects.playAudio(trimmed, audio.sampleRate);
*/